import torch
import numpy as np

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field

def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    coarse_model_secondary_list,
    fine_model_secondary_list,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
    )

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
    )

    rgb_coarse_secondary_list = []
    len_coarse_model_secondary_list = len(coarse_model_secondary_list)
    for i in range(len_coarse_model_secondary_list):
        radiance_field_coarse_secondary = run_network(
            coarse_model_secondary_list[i],
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )

        (
            rgb_coarse_secondary,
            disp_coarse_secondary,
            acc_coarse_secondary,
            weights_secondary,
            depth_coarse_secondary,
        ) = volume_render_radiance_field(
            radiance_field_coarse_secondary,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )
        rgb_coarse_secondary_list.append(rgb_coarse_secondary)
    
    # print("rgb_coarse_secondary_list: \n", rgb_coarse_secondary_list)


    rgb_fine_secondary_list = []
    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )


        len_fine_model_secondary_list = len(fine_model_secondary_list)
        for i in range(len_fine_model_secondary_list):
            radiance_field_fine_secondary = run_network(
                fine_model_secondary_list[i],
                pts,
                ray_batch,
                getattr(options.nerf, mode).chunksize,
                encode_position_fn,
                encode_direction_fn,
            )

            rgb_fine_secondary, disp_fine_secondary, acc_fine_secondary, _, _ = volume_render_radiance_field(
                radiance_field_fine_secondary,
                z_vals,
                rd,
                radiance_field_noise_std=getattr(
                    options.nerf, mode
                ).radiance_field_noise_std,
                white_background=getattr(options.nerf, mode).white_background,
            )

            rgb_fine_secondary_list.append(rgb_fine_secondary)


    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, rgb_coarse_secondary_list, rgb_fine_secondary_list


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    coarse_model_secondary_list,
    fine_model_secondary_list,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            coarse_model_secondary_list,
            fine_model_secondary_list,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        for batch in batches
    ]
    synthesized_images_ = list(zip(*pred))

    # print("synthesized_images_: ", len(synthesized_images_))
    # print("len(synthesized_images_[0]): ", len(synthesized_images_[0])) # (800*800) / 80000 = 8 (tuple)
    # print("len(synthesized_images_[0][0]): ", len(synthesized_images_[0][0]))   # 80000 (torch.Tensor)
    # print("len(synthesized_images_[0][0][0]): ", len(synthesized_images_[0][0][0])) # 3 (RGB) (torch.Tensor)
    # print("len(synthesized_images_[7][0][-1]): ", len(synthesized_images_[7][0][-1]))
    # print("synthesized_images_[7]: ", synthesized_images_[7])
    
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images_[:6]
    ]
    
    # print("synthesized_images size(): ", synthesized_images[0].size())  # torch.Size([640000, 3])
    # print("len(synthesized_images): ", len(synthesized_images))     # 6

    # print("len(synthesized_images_[6]): ", len(synthesized_images_[6])) # (800*800) / 80000 = 8 (tuple)
    # print("len(synthesized_images_[6][0]): ", len(synthesized_images_[6][0]))    # 3 (depends on number of secondary models) (list)
    # print("len(synthesized_images_[6][0][0]): ", len(synthesized_images_[6][0][0]))    # 80000  (torch.Tensor)
    # print("len(synthesized_images_[6][0][0][0]): ", len(synthesized_images_[6][0][0][0]))   # 3 (RGB) (torch.Tensor)
    # for i in range(len(coarse_model_secondary_list)):
    #     synthesized_images_coarse_secondary = [
    #         torch.cat([image], dim=0) if image[0] is not None else None
    #         for image in synthesized_images_[6][i]
    #     ]

    synthesized_images_coarse_secondary = []

    # Iterate through the secondary coarse models (number of secondary coarse models, 3 in this case)
    for j in range(len(synthesized_images_[-2][0])):
        images_per_model = []  # List to store images for each secondary coarse model
        
        # Iterate through the outermost dimension ((800x800)/80000 = 8 in this case, 80000 is the chunksize)
        for i in range(len(synthesized_images_[-2])):
            # Extract a specific slice of data from synthesized_images_[6]
            slice_of_data = synthesized_images_[-2][i][j]  # This will have dimensions 80000x3
            
            # Append the slice_of_data to images_per_model
            images_per_model.append(slice_of_data)
        
        # print("images_per_model: ", type(images_per_model)) # <class 'list'>
        # print("images_per_model: ", len(images_per_model)) # 8
        # k = 0
        # for k in range(len(images_per_model)):
        #     print("i: {}   |   images_per_model size(): {}".format(k, images_per_model[k].size()))
        #     k += 1

        # Combine the slices of data into a single tensor
        combined_data = torch.cat(images_per_model, dim=0) if images_per_model[0] is not None else None
        
        # Append the combined data to synthesized_images_coarse_secondary
        synthesized_images_coarse_secondary.append(combined_data)
    
    # print("len(synthesized_images_coarse_secondary): ", len(synthesized_images_coarse_secondary))   # 3 (depends on number of secondary models) (<class 'list'>)
    # print("synthesized_images_coarse_secondary[0].size(): ", synthesized_images_coarse_secondary[0].size()) # torch.Size([640000, 3])
    

    synthesized_images_fine_secondary = []
    for j in range(len(synthesized_images_[-1][0])):
        images_per_model = []  # List to store images for each secondary fine model
        for i in range(len(synthesized_images_[-1])):
            slice_of_data = synthesized_images_[-1][i][j]
            images_per_model.append(slice_of_data)
        combined_data = torch.cat(images_per_model, dim=0) if images_per_model[0] is not None else None
        synthesized_images_fine_secondary.append(combined_data)
    
    # print("len(synthesized_images_fine_secondary): ", len(synthesized_images_fine_secondary))   # 3 (depends on number of secondary models) (<class 'list'>)
    # print("synthesized_images_fine_secondary[0].size(): ", synthesized_images_fine_secondary[0].size()) # torch.Size([640000, 3])
    
    if mode == "validation":
        # print("synthesized_images: \n", synthesized_images)
        # print("len(synthesized_images): ", len(synthesized_images)) # 6   (<class 'list'>)
        # print("synthesized_images[0].size(): ", synthesized_images[0].size()) # torch.Size([640000, 3]) (<class torch.Tensor>)

        # print("np.shape (restore_shapes): ", np.shape(np.array(restore_shapes)))  # (6,) <class 'list'>
        # print("restore_shapes: ", restore_shapes) # [torch.Size([800, 800, 3]), torch.Size([800, 800]), torch.Size([800, 800]), torch.Size([800, 800, 3]), torch.Size([800, 800]), torch.Size([800, 800])]

        synthesized_images_valid = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # print("synthesized_images_valid: ", type(synthesized_images_valid))   # <class 'list'>
        # print("len(synthesized_images_valid): ", len(synthesized_images_valid)) # 6
        # print("type(synthesized_images_valid[0]): ", type(synthesized_images_valid[0])) # <torch.Tensor>
        # print("synthesized_images_valid[0].size(): ", synthesized_images_valid[0].size())   # torch.Size([800, 800, 3])

        # TO-DO: Generate the RESHAPE ([800, 800, 3]) synthesized images for coarse and fine secondary models
        restore_shapes_secondary = []
        for i in range(len(synthesized_images_coarse_secondary)):
            restore_shapes_secondary.append(restore_shapes[0])
        # print("restore_shapes_secondary: ", restore_shapes_secondary)   # [torch.Size([800, 800, 3]), torch.Size([800, 800, 3]), torch.Size([800, 800, 3])]

        synthesized_images_coarse_secondary_valid = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images_coarse_secondary, restore_shapes_secondary)
        ]

        # print("len(synthesized_images_coarse_secondary_valid): ", len(synthesized_images_coarse_secondary_valid))   # 3 (<class 'list'>)
        # print("synthesized_images_coarse_secondary_valid[0].size(): ", synthesized_images_coarse_secondary_valid[0].size())   # torch.Size([800, 800, 3]) <class 'torch.Tensor'>

        synthesized_images_fine_secondary_valid = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images_fine_secondary, restore_shapes_secondary)
        ]



        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            synthesized_images_valid.append(synthesized_images_coarse_secondary_valid)
            synthesized_images_valid.append(synthesized_images_fine_secondary_valid)

            #print("tuple(synthesized_images): \n", tuple(synthesized_images))
            return tuple(synthesized_images_valid)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images_valid + [None, None, None])
    

    synthesized_images.append(synthesized_images_coarse_secondary)
    synthesized_images.append(synthesized_images_fine_secondary)

    return tuple(synthesized_images)

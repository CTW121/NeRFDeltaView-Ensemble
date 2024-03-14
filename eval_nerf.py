# https://github.com/krrish94/nerf-pytorch

import argparse
from logging import config
import os
import time
import json

import imageio
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    # load_blender_data,
    # load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
)


def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def get_render_image(vector_magnitude, rotation_matrix, focal_length, iteration):

    print("Generating rendered image using NeRF model in PyTorch.")

    config_folder = 'config'
    config_file = 'config.yml'
    checkpoint_file = 'checkpoint{}.ckpt'.format(iteration-1)

    config_file_path = os.path.join(config_folder, config_file)
    checkpoint_file_path = os.path.join(config_folder, checkpoint_file)


    cfg = None
    with open(config_file_path, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    # print("model_coarse: \n", model_coarse)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)
    # print("model_fine: \n", model_fine)

    # *** Secondary coarse models *** #
    coarse_model_secondary_list = []
    num_models = cfg.experiment.num_models_secondary
    for i in range(num_models):
        model_coarse_secondary = getattr(models, cfg.models_secondary.coarse.type)(
            num_layers=cfg.models_secondary.coarse.num_layers,
            hidden_size=cfg.models_secondary.coarse.hidden_size,
            skip_connect_every=cfg.models_secondary.coarse.skip_connect_every,
            num_encoding_fn_xyz=cfg.models_secondary.coarse.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models_secondary.coarse.num_encoding_fn_dir,
            include_input_xyz=cfg.models_secondary.coarse.include_input_xyz,
            include_input_dir=cfg.models_secondary.coarse.include_input_dir,
            use_viewdirs=cfg.models_secondary.coarse.use_viewdirs,
        )
        coarse_model_secondary_list.append(model_coarse_secondary)
        coarse_model_secondary_list[i].to(device)
    
    # *** Secondary fine models *** #
    fine_model_secondary_list = []
    if hasattr(cfg.models_secondary, "fine"):
        for i in range(num_models):
            model_fine_secondary = getattr(models, cfg.models_secondary.fine.type)(
                num_layers=cfg.models_secondary.fine.num_layers,
                hidden_size=cfg.models_secondary.fine.hidden_size,
                skip_connect_every=cfg.models_secondary.fine.skip_connect_every,
                num_encoding_fn_xyz=cfg.models_secondary.fine.num_encoding_fn_xyz,
                num_encoding_fn_dir=cfg.models_secondary.fine.num_encoding_fn_dir,
                include_input_xyz=cfg.models_secondary.fine.include_input_xyz,
                include_input_dir=cfg.models_secondary.fine.include_input_dir,
                use_viewdirs=cfg.models_secondary.fine.use_viewdirs,
            )
            fine_model_secondary_list.append(model_fine_secondary)
            fine_model_secondary_list[i].to(device)
    
    # checkpoint = torch.load(configargs.checkpoint)
    checkpoint = torch.load(checkpoint_file_path)

    #print(checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    
    for i, coarse_model_secondary in enumerate(coarse_model_secondary_list):
        coarse_model_secondary.load_state_dict(checkpoint["model_coarse_secondary_state_dict"][i])
        
    for i, fine_model_secondary in enumerate(fine_model_secondary_list):
        fine_model_secondary.load_state_dict(checkpoint["model_fine_secondary_state_dict"][i])

    # if "height" in checkpoint.keys():
    #     hwf[0] = checkpoint["height"]
    # if "width" in checkpoint.keys():
    #     hwf[1] = checkpoint["width"]
    # if "focal_length" in checkpoint.keys():
    #     hwf[2] = checkpoint["focal_length"]

    # ========= #
    # theta: longitude (east / west)
    # phi: latitude (north / south)
    # focal_length: depth (lower value->further; higher value->nearer)

    # theta = 0
    # phi = 0 
    focal_length = 1200
    # ========= #

    hwf = []

    hwf.append(800)
    hwf.append(800)

    # focal length (change the zooming of the scene; zoom in or zoom out)
    # higher focal length value -> zoom in
    # lower focal length value -> zoom out
    hwf.append(focal_length)

    # print("hwf[0]: ", hwf[0])   # hwf[0]:  800
    # print("hwf[1]: ", hwf[1])   # hwf[1]:  800
    # print("hwf[2]: ", hwf[2])   # hwf[2]:  1111.1110311937682

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    def translate_by_t_along_z(t):
        tform = np.eye(4).astype(np.float32)
        tform[2][3] = t
        return tform


    def rotate_by_phi_along_x(phi):
        tform = np.eye(4).astype(np.float32)
        tform[1, 1] = tform[2, 2] = np.cos(phi)
        tform[1, 2] = -np.sin(phi)
        tform[2, 1] = -tform[1, 2]
        return tform


    def rotate_by_theta_along_y(theta):
        tform = np.eye(4).astype(np.float32)
        tform[0, 0] = tform[2, 2] = np.cos(theta)
        tform[0, 2] = -np.sin(theta)
        tform[2, 0] = -tform[0, 2]
        return tform

    def pose_spherical(theta, phi, radius):
        c2w = translate_by_t_along_z(radius)
        c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
        c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w

    translationMatrix = np.eye(4).astype(np.float32)
    # translationMatrix[2, 3] = vector_magnitude
    translationMatrix[2, 3] = 4.0 if vector_magnitude >= 4.0 else vector_magnitude
    # translationMatrix = np.array([[1.0, 0.0, 0.0, 0.0],
    #                        [0.0, 1.0, 0.0, 0.0],
    #                        [0.0, 0.0, 1.0, vector_magnitude],
    #                        [0.0, 0.0, 0.0, 1.0]])
    rotationMatrix = np.linalg.inv(rotation_matrix)
    # tranformationMatrix = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotationMatrix @ translationMatrix
    # tranformationMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ rotationMatrix @ translationMatrix
    tranformationMatrix = rotate_by_phi_along_x(-90 / 180.0 * np.pi) @ np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ rotationMatrix @ translationMatrix

    render_poses = torch.stack(
        [
            torch.from_numpy(tranformationMatrix)
        ],
        0,
    )

    render_poses = render_poses.float().to(device)

    # Create directory to save images to.
    # os.makedirs(configargs.savedir, exist_ok=True)
    save_dir = '.'

    # if configargs.save_disparity_image:
    #     os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)

    # Evaluation loop
    times_per_image = []
    for i, pose in enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = pose[:3, :4]
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            # print("ray_origins.shape: ", ray_origins.shape)
            # print("ray_directions: ", ray_directions.shape)
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _, rgb_coarse_secondary, rgb_fine_secondary \
            = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                coarse_model_secondary_list,
                fine_model_secondary_list,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            # if configargs.save_disparity_image:
            #     disp = disp_fine if disp_fine is not None else disp_coarse
        times_per_image.append(time.time() - start)
        # if configargs.savedir:
        if save_dir:
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            savefile = f"{i:04d}.png"
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
            )
            # if configargs.save_disparity_image:
            #     savefile = os.path.join(configargs.savedir, "disparity", f"{i:04d}.png")
            #     imageio.imwrite(savefile, cast_to_disparity_image(disp))
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")
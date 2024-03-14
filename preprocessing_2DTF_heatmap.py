import vtk
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import os
import time
from datetime import datetime

from helpers import helpers


def similarity_vectors(vector1, vector2):
    """
    Calculate the angle in degrees between two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cos_similarity = dot_product / (norm_vector1 * norm_vector2)

    # Calculate the angle in degrees
    angle = np.arccos(cos_similarity) * 180.0 / np.pi

    return angle, cos_similarity


def main():

    colors = vtk.vtkNamedColors()

    dataset = "chair"            # dataset: lego / hotdog / chair
    dataset_size = "full"       # full or partial
    iterations = 200000

    data_folder = "data"

    opacity_file_name = "{}_{}_{}_opacity.vtk".format(dataset, dataset_size, iterations)
    opacity_file_path = os.path.join(data_folder, opacity_file_name)
    opacity_volume, opacity_reader = helpers.vtk_read_volume_from_file(opacity_file_path)

    color_uncertainty_file_name = "{}_{}_{}_uncertainty_color.vtk".format(dataset, dataset_size, iterations)
    color_uncertainty_file_path = os.path.join(data_folder, color_uncertainty_file_name)
    color_uncertainty_volume, color_uncertainty_reader = helpers.vtk_read_volume_from_file(color_uncertainty_file_path)

    density_uncertainty_file_name = "{}_{}_{}_uncertainty_density.vtk".format(dataset, dataset_size, iterations)
    density_uncertainty_file_path = os.path.join(data_folder, density_uncertainty_file_name)
    density_uncertainty_volume, density_uncertainty_reader = helpers.vtk_read_volume_from_file(density_uncertainty_file_path)

    """
    Isosurface
    """
    isosurface_value = 0.90

    contour_opacity = helpers.vtk_contour_filter(opacity_volume, filter_value=isosurface_value)
    mapper_opacity = helpers.vtk_poly_data_mapper(contour_opacity)
    actor_opacity = helpers.vtk_create_actor(mapper_opacity, 'DodgerBlue')

    """
    Color uncertainty
    """
    opacity_tf_color_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.80, 0.00], [0.85, 1.00]])
    alpha_tf_color_uncertainty = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 0.0, 0.0, 0.0], [1.00, 1.0, 0.0, 0.0]])
    volume_property_color_uncertainty = helpers.vtk_volume_property(alpha_tf_color_uncertainty, opacity_tf_color_uncertainty)
    volume_mapper_color_uncertainty = helpers.vtk_volume_ray_cast_mapper(color_uncertainty_reader)
    volume_color_uncertainty = helpers.vtk_create_volume(volume_mapper_color_uncertainty, volume_property_color_uncertainty)

    """
    Density uncertainty
    """
    # opacity_tf_density_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.10, 0.00], [0.20, 1.00]])
    # opacity_tf_density_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.20, 1.00]])
    opacity_tf_density_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.20, 0.00], [0.60, 1.00]])
    alpha_tf_density_uncertainty = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 0.0, 0.0, 0.0], [1.00, 1.0, 0.0, 0.0]])
    volume_property_density_uncertainty = helpers.vtk_volume_property(alpha_tf_density_uncertainty, opacity_tf_density_uncertainty)
    volume_mapper_density_uncertainty = helpers.vtk_volume_ray_cast_mapper(density_uncertainty_reader)
    volume_density_uncertainty = helpers.vtk_create_volume(volume_mapper_density_uncertainty, volume_property_density_uncertainty)

    """
    Renderers
    """
    # isosurface
    interactor_isosurface = vtk.vtkRenderWindowInteractor()
    render_window_isosurface = vtk.vtkRenderWindow()
    renderer_isosurface = vtk.vtkRenderer()

    render_window_isosurface.AddRenderer(renderer_isosurface)
    interactor_isosurface.SetRenderWindow(render_window_isosurface)

    renderer_isosurface.AddVolume(actor_opacity)
    renderer_isosurface.SetBackground(colors.GetColor3d('White'))

    # uncertainty color
    interactor_color = vtk.vtkRenderWindowInteractor()
    render_window_color = vtk.vtkRenderWindow()
    renderer_color = vtk.vtkRenderer()

    render_window_color.AddRenderer(renderer_color)
    interactor_color.SetRenderWindow(render_window_color)

    renderer_color.AddVolume(volume_color_uncertainty)
    renderer_color.SetBackground(colors.GetColor3d('Black'))

    # uncertainty density
    interactor_density = vtk.vtkRenderWindowInteractor()
    render_window_density = vtk.vtkRenderWindow()
    renderer_density = vtk.vtkRenderer()

    render_window_density.AddRenderer(renderer_density)
    interactor_density.SetRenderWindow(render_window_density)

    renderer_density.AddVolume(volume_density_uncertainty)
    renderer_density.SetBackground(colors.GetColor3d('Black'))


    camera = renderer_isosurface.GetActiveCamera()
    original_orient = helpers.vtk_get_orientation(renderer_isosurface)
    print("original_orient: ", original_orient["orientation"])

    # default_view_up = [0.0, 0.0, 1.0]
    # camera.SetViewUp(default_view_up)

    # camera.SetPosition(0.0, 0.0, 0.0)
    # camera.SetFocalPoint(0.0, 1.0, 0.0)
    # camera.SetViewUp([0.0, 1.0, 0.0])

    # azimuth = [0, 45, 90]
    # elevation = [0, 45, 90]
    azimuth = [i for i in range(0, 360+1, 15)]    # east-west
    elevation = [i for i in range(0, 360+1, 15)]  # north-south
    azimuth_len = len(azimuth)
    elevation_len = len(elevation)
    # print("azimuth: ", azimuth)
    # print("elevation: ", elevation)

    means_color = np.zeros((azimuth_len, elevation_len))
    standard_deviations_color = np.zeros((azimuth_len, elevation_len))

    means_density = np.zeros((azimuth_len, elevation_len))
    standard_deviations_density = np.zeros((azimuth_len, elevation_len))

    z_buffer_data_isosurface = vtk.vtkFloatArray()
    z_buffer_data_color_uncertainty = vtk.vtkFloatArray()
    z_buffer_data_density_uncertainty = vtk.vtkFloatArray()

    for i in range(azimuth_len):
        for j in range(elevation_len):
            helpers.vtk_set_orientation(renderer_isosurface, original_orient)
            helpers.vtk_set_orientation(renderer_color, original_orient)
            helpers.vtk_set_orientation(renderer_density, original_orient)

            camera.Azimuth(azimuth[i]) # east-west
            camera.Elevation(elevation[j]) # north-south

            # https://discourse.vtk.org/t/vtkrenderer-error/6143/2
            view_up_vector = camera.GetViewUp()
            view_plane_normal = camera.GetViewPlaneNormal()
            # print("view_up_vector: ", view_up_vector)
            # print("view_plane_normal: ", view_plane_normal)
            # print("angle", angle_between_vectors(view_up_vector, view_plane_normal))

            angle, cos_similarity = similarity_vectors(view_up_vector, view_plane_normal)
            # print("azimuth: {}  | elevation: {} | angle: {:.2f} | cosine similarity: {:.2f}".format(azimuth[i], elevation[j], angle, cos_similarity))

            if abs(cos_similarity) > 0.95:
                camera.SetViewUp(0.0, 0.0, 1.0)
            else:
                camera.SetViewUp(0.0, 1.0, 0.0)
                
            renderer_isosurface.ResetCamera()
            renderer_color.SetActiveCamera(camera)
            renderer_color.ResetCamera()
            renderer_density.SetActiveCamera(camera)
            renderer_density.ResetCamera()

            # === Z-buffer === #
            renderer_isosurface.PreserveDepthBufferOff()
            renderer_isosurface.GetRenderWindow().Render()

            renderer_color.PreserveDepthBufferOff()
            renderer_color.GetRenderWindow().Render()

            renderer_density.PreserveDepthBufferOff()
            renderer_density.GetRenderWindow().Render()

            xmax_isosurface, ymax_isosurface = renderer_isosurface.GetRenderWindow().GetActualSize()
            renderer_isosurface.GetRenderWindow().GetZbufferData(0, 0, ymax_isosurface-1, xmax_isosurface-1, z_buffer_data_isosurface)

            xmax_color, ymax_color = renderer_color.GetRenderWindow().GetActualSize()
            renderer_color.PreserveDepthBufferOn()
            renderer_color.GetRenderWindow().GetZbufferData(0, 0, ymax_color-1, xmax_color-1, z_buffer_data_color_uncertainty)
            renderer_color.GetRenderWindow().SetZbufferData(0, 0, ymax_color-1, xmax_color-1, z_buffer_data_isosurface)

            xmax_density, ymax_density = renderer_density.GetRenderWindow().GetActualSize()
            renderer_density.PreserveDepthBufferOn()
            renderer_density.GetRenderWindow().GetZbufferData(0, 0, ymax_density-1, xmax_density-1, z_buffer_data_density_uncertainty)
            renderer_density.GetRenderWindow().SetZbufferData(0, 0, ymax_density-1, xmax_density-1, z_buffer_data_isosurface)

            # renderer_isosurface.GetRenderWindow().Render()
            # renderer_color.GetRenderWindow().Render()
            # renderer_density.GetRenderWindow().Render()
            # ================ #

            render_window_isosurface.Render()
            render_window_color.Render()
            render_window_density.Render()

            mean_color, standard_deviation_color = helpers.mean_standard_deviation(azimuth_len, elevation_len, render_window_color)
            mean_density, standard_deviation_density = helpers.mean_standard_deviation(azimuth_len, elevation_len, render_window_density)

            means_color[i][j] = mean_color
            standard_deviations_color[i][j] = standard_deviation_color
            
            means_density[i][j] = mean_density
            standard_deviations_density[i][j] = standard_deviation_density

    # print("means_color: ", means_color)
    # print("standard_deviations_color: ", standard_deviations_color)

    # Define the file paths including the folder
    color_means_file = os.path.join(data_folder, "color_means.csv")
    color_stddev_file = os.path.join(data_folder, "color_standard_deviations.csv")
    
    density_means_file = os.path.join(data_folder, "density_means.csv")
    density_stddev_file = os.path.join(data_folder, "density_standard_deviations.csv")

    # Save the CSV files with the specified paths
    np.savetxt(color_means_file, means_color, delimiter=",")
    np.savetxt(color_stddev_file, standard_deviations_color, delimiter=",")
    
    np.savetxt(density_means_file, means_density, delimiter=",")
    np.savetxt(density_stddev_file, standard_deviations_density, delimiter=",")

    # interactor_isosurface.Start()
    # interactor_color.Start()
    # interactor_density.Start()


if __name__ == "__main__":
    start_time = datetime.now()
    starting_time = start_time.strftime("%H:%M:%S")
    print("Start time: ", starting_time)

    main()

    end_time = datetime.now()
    ending_time = end_time.strftime("%H:%M:%S")
    print("End time: ", ending_time)

    print('Duration: {}'.format(end_time - start_time))
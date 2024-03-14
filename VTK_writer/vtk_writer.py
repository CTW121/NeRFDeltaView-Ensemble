#!/usr/bin/env python

"""
Create a ImageData grid and then assign an uncertainty scalar value and opacity scalar value to each point.
"""

import torch

import argparse
import sys
import os
import time
from datetime import datetime

import vtk

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkImagingCore import vtkImageCast
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkMath, vtkFloatArray
from vtkmodules.vtkIOLegacy import (
    vtkStructuredPointsReader,
    vtkStructuredPointsWriter,
    vtkRectilinearGridWriter,
)
import vtk.util.numpy_support as numpy_support
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLRayCastImageDisplayHelper, vtkOpenGLGPUVolumeRayCastMapper
from vtkmodules.vtkCommonTransforms import (
    vtkTransform,
)
from vtkmodules.vtkFiltersGeneral import (
    vtkTransformFilter,
    vtkTransformPolyDataFilter,
)
from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
    vtkPiecewiseFunction,
    vtkRectilinearGrid,
    vtkImageData,
    vtkStructuredPoints,
    vtkPlane,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkColorTransferFunction,
    vtkVolumeProperty,
    vtkVolume,
)
from vtkmodules.vtkFiltersGeometry import (
    vtkImageDataGeometryFilter,
    vtkRectilinearGridGeometryFilter,
)
from vtkmodules.vtkIOXML import (
    vtkXMLImageDataReader,
    vtkXMLImageDataWriter,
)
from vtkmodules.vtkIOLegacy import (
    vtkUnstructuredGridWriter,
    #vtkImageDataWriter,
)
from vtkmodules.vtkFiltersCore import (
    vtkContourFilter,
    vtkCutter,
    vtkPolyDataNormals,
    vtkStripper,
    vtkStructuredGridOutlineFilter,
    vtkTubeFilter,
    vtkClipPolyData,
)

import numpy as np
import torch
import random
import math
import yaml
#import simplejson

from scipy.spatial.distance import pdist, squareform

from nerf import (CfgNode, models, helpers)

def main():
    ### *** user defined parameters for vtkImageData *** ###

    scene = 'chair'
    dataset = 'partial'    # dataset size: 'full' or 'partial'/'selected' view dataset
    iteration = 100000

    xyzMin = -1.0
    xyzMax = 1.0
    xyzNumPoint = 100
    radiance_field_noise_std = 0.2
    
    ### *** user defined parameters for vtkImageData *** ###

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # clear memory in GPU CUDA
    torch.cuda.empty_cache()

    model_fine = models.FlexibleNeRFModel(
        cfg.models.fine.num_layers,
        cfg.models.fine.hidden_size,
        cfg.models.fine.skip_connect_every,
        cfg.models.fine.num_encoding_fn_xyz,
        cfg.models.fine.num_encoding_fn_dir,
    )

    fine_model_secondary_list = []
    for i in range(cfg.experiment.num_models_secondary):
        model_fine_secondary = models.FlexibleNeRFModel(
            cfg.models_secondary.fine.num_layers,
            cfg.models_secondary.fine.hidden_size,
            cfg.models_secondary.fine.skip_connect_every,
            cfg.models_secondary.fine.num_encoding_fn_xyz,
            cfg.models_secondary.fine.num_encoding_fn_dir,
        )
        fine_model_secondary_list.append(model_fine_secondary)

    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        
        for i, fine_model_secondary in enumerate(fine_model_secondary_list):
            fine_model_secondary.load_state_dict(checkpoint["model_fine_secondary_state_dict"][i])
    else:
        sys.exit("Please enter the path of the checkpoint file.")
    
    model_fine.eval()
    for i in range(cfg.experiment.num_models_secondary):
        fine_model_secondary_list[i].eval()
    
    include_input_xyz = 3 if cfg.models.fine.include_input_xyz else 0
    include_input_dir = 3 if cfg.models.fine.include_input_dir else 0
    dim_xyz = include_input_xyz + 2 * 3 * cfg.models.fine.num_encoding_fn_xyz
    dim_dir = include_input_dir + 2 * 3 * cfg.models.fine.num_encoding_fn_dir
    if not cfg.models.fine.use_viewdirs:
        dim_dir = 0

    print("Scene: ", scene)
    print("Dataset: ", dataset)
    print("Iteration: ", iteration)
    print("Points per dimension: ", xyzNumPoint)

    dxyz = abs(xyzMax - xyzMin) / (xyzNumPoint - 1)

    voxelVol = vtkStructuredPoints()
    #voxelVol = vtkImageData()
    voxelVol.SetDimensions(xyzNumPoint, xyzNumPoint, xyzNumPoint)
    voxelVol.SetOrigin(-(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2))
    # voxelVol.SetOrigin(-(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2), 0.0)   # <===
    # voxelVol.SetOrigin(-(abs(xyzMax-xyzMin)/2), -(abs(xyzMax-xyzMin)/2), -0.5)   # <===
    voxelVol.SetSpacing(dxyz, dxyz, dxyz)

    uncertaintyColor = []
    uncertaintyDensity = []

    #arrayColor = vtkDoubleArray()
    arrayColor = vtkFloatArray()
    arrayColor.SetNumberOfComponents(3) # this is 3 for a vector
    arrayColor.SetNumberOfTuples(voxelVol.GetNumberOfPoints())

    #arrayDensity = vtkDoubleArray()
    arrayDensity = vtkFloatArray()
    arrayDensity.SetNumberOfComponents(1) # this is 3 for a vector
    arrayDensity.SetNumberOfTuples(voxelVol.GetNumberOfPoints())

    tensor_input = torch.zeros(1, dim_xyz+dim_dir)
    npoints = voxelVol.GetNumberOfPoints()

    #densityTensor = torch.zeros(1, npoints)

    for i in range(npoints):
        if i%100000 == 0:
            print("i: ", i)
        x, y, z = voxelVol.GetPoint(i)
        xyz  = [x, y, z]
        xyz_tensor = torch.Tensor([xyz])

        encode_pos = helpers.embedding_encoding(xyz_tensor, num_encoding_functions=cfg.models.fine.num_encoding_fn_xyz)
        tensor_input[..., : dim_xyz] = encode_pos

        if cfg.models.fine.use_viewdirs:
                encode_dir = helpers.embedding_encoding(xyz_tensor, num_encoding_functions=cfg.models.fine.num_encoding_fn_dir)
                # tensor_input[..., dim_xyz :] = encode_dir
                encode_dir_zeros = torch.zeros_like(encode_dir)
                tensor_input[..., dim_xyz :] = encode_dir_zeros
        
        output = model_fine(tensor_input)
        output_list = output.tolist()[0]     # [R, G, B, sigma] ###

        arrayDensity.SetValue(i, output_list[3])
        #densityTensor[..., i] = output[..., 3]

        # === check === #
        # print("{}: {}   |   {}".format(i, xyz, output_list[3]))
        # print("encode_dir: ", encode_dir)
        # print("tensor_input: ", tensor_input)
        # print("output: ", output)
        # print("output_list: ", output_list)
        # === check === #

        ### === Uncertainty for RGB color and density === ###
        output_point_color = []
        output_point_density = []

        color = torch.sigmoid(output[:, :3])
        output_point_color.append(color.tolist()[0])    # Append the RGB color from the primary model into the list

        sigma = torch.nn.functional.relu(output[:, 3])
        output_point_density.append([sigma.tolist()[0]])    # Append the density from the primary model into the list

        for i in range(cfg.experiment.num_models_secondary):
            output_secondary = fine_model_secondary_list[i](tensor_input)       # [R, G, B, sigma]
            color = torch.sigmoid(output_secondary[:, :3])
            output_point_color.append(color.tolist()[0])    # Append the RGB color from the secondary models into the list
            sigma = torch.nn.functional.relu(output_secondary[:,3])
            output_point_density.append([sigma.tolist()[0]])    # Append the density from the secondary models into the list
        
        distancesColor = pdist(output_point_color, metric='euclidean')
        average_distance_color = np.mean(distancesColor)
        uncertaintyColor.append(average_distance_color)

        distanceDensity = pdist(output_point_density, metric='euclidean')
        average_distance_density = np.mean(distanceDensity)
        uncertaintyDensity.append(average_distance_density)
    
    
    ### === Generate VTK file for opacity === ###
    if radiance_field_noise_std > 0.0:
        noise = np.random.randn(npoints)
        #noise = torch.randn(npoints)
    else:
        noise = np.zeros_like(npoints)
    arraySigma_tensor = torch.nn.functional.relu(torch.Tensor(arrayDensity + noise))
    #arraySigma_tensor = torch.nn.functional.relu(densityTensor + noise)
    alpha_tensor = 1.0 - torch.exp(-arraySigma_tensor)
    alpha = numpy_support.numpy_to_vtk(num_array=alpha_tensor.numpy(), deep=True)

    # === check === #
    # print(alpha_tensor)
    # === check === #

    voxelVolOpacity = vtkStructuredPoints()
    voxelVolOpacity.DeepCopy(voxelVol)
    voxelVolOpacity.GetPointData().SetScalars(alpha)

    writerOpacity = vtkStructuredPointsWriter()
    writerOpacity.WriteExtentOn()
    writerOpacity.SetFileName("{}_{}_{}_opacity.vtk".format(scene, dataset, iteration))
    writerOpacity.SetInputData(voxelVolOpacity)
    #writerOpacity.SetFileTypeToBinary()
    writerOpacity.Write()

    ### === Generate VTK file for RGB color uncertainty === ###
    uncertaintyColor_np = np.array(uncertaintyColor)
    uncertaintyColor_np_normalize = (uncertaintyColor_np - np.min(uncertaintyColor_np)) / (np.max(uncertaintyColor_np) - np.min(uncertaintyColor_np))
    arrayUncertaintyColor_vtk = numpy_support.numpy_to_vtk(num_array=uncertaintyColor_np_normalize, deep=True)

    voxelVolUncertaintyColor = vtkStructuredPoints()
    voxelVolUncertaintyColor.DeepCopy(voxelVol)
    voxelVolUncertaintyColor.GetPointData().SetScalars(arrayUncertaintyColor_vtk)

    writerUncertaintyColor = vtkStructuredPointsWriter()
    writerUncertaintyColor.WriteExtentOn()
    writerUncertaintyColor.SetFileName("{}_{}_{}_uncertainty_color.vtk".format(scene, dataset, iteration))
    writerUncertaintyColor.SetInputData(voxelVolUncertaintyColor)
    writerUncertaintyColor.Write()

    ### === Generate VTK file for density uncertainty === ###
    uncertaintyDesnity_np = np.array(uncertaintyDensity)
    uncertaintyDesnity_np_normalize = (uncertaintyDesnity_np - np.min(uncertaintyDesnity_np)) / (np.max(uncertaintyDesnity_np) - np.min(uncertaintyDesnity_np))
    arrayUncertaintyDensity_vtk = numpy_support.numpy_to_vtk(num_array=uncertaintyDesnity_np_normalize, deep=True)

    voxelVolUncertaintyDensity = vtkStructuredPoints()
    voxelVolUncertaintyDensity.DeepCopy(voxelVol)
    voxelVolUncertaintyDensity.GetPointData().SetScalars(arrayUncertaintyDensity_vtk)

    writerUncertaintyDensity = vtkStructuredPointsWriter()
    writerUncertaintyDensity.WriteExtentOn()
    writerUncertaintyDensity.SetFileName("{}_{}_{}_uncertainty_density.vtk".format(scene, dataset, iteration))
    writerUncertaintyDensity.SetInputData(voxelVolUncertaintyDensity)
    writerUncertaintyDensity.Write()


if __name__ == "__main__":
    start_time = datetime.now()
    starting_time = start_time.strftime("%H:%M:%S")
    print("Start time: ", starting_time)

    main()
    
    end_time = datetime.now()
    ending_time = end_time.strftime("%H:%M:%S")
    print("End time: ", ending_time)

    print('Duration: {}'.format(end_time - start_time))
# NeRFDeltaView Ensemble

We developed a visualization tool for visualizing the uncertainty estimated by the [NeRF Uncertainty Ensemble](https://github.com/CTW121/NeRF-Uncertainty-Ensemble).

Uncertainty visualization provides users with an in-depth understanding of the data for analysis and to perform confident and informed decision-making. The main purpose of our tool is to highlight the significance of interactive visualization in enabling users to explore the estimated uncertainty in synthesized scenes, identify model limitations, and aid in understanding NeRF model uncertainty.

## Prerequisites

Ensure you have met the following requirements:
- You have a **Linux/Windows/Mac** machine.
- You have installed **Python 3.8 or higher**.
- You have installed [**PyTorch**](https://pytorch.org/).
- You have installed [**PyQt6**](https://doc.qt.io/qtforpython-6/).
- You have installed [**VTK from Python wrappers**](https://docs.vtk.org/en/latest/getting_started/index.html).

We recommend running the application using [**conda**](https://docs.conda.io/en/latest/).

## How to launch the Visualization Tool Application?

To launch the NeRFDeltaView Ensemble visualization tool application, follow these steps:

1. Copy the trained model (checkpoint and yml files) from [NeRF Uncertainty Ensemble](https://github.com/CTW121/NeRF-Uncertainty-Ensemble) to the [VTK_writer](https://github.com/CTW121/NeRFDeltaView-Ensemble/tree/master/VTK_writer) folder.

2. Within the [VTK_writer](https://github.com/CTW121/NeRFDeltaView-Ensemble/tree/master/VTK_writer) folder, execute `python vtk_writer.py` to generate the VTK 3D volumetric data files (estimated opacity, color, and density). Then, copy these VTK 3D volumetric data files to the [data](https://github.com/CTW121/NeRFDeltaView-Ensemble/tree/master/data) folder.

3. Run `python preprocessing_2DTF_heatmap.py` to produce CSV files containing color and density means, as well as standard deviations for heatmap visualization.

4. Run `python NeRFDeltaView.py` to launch the visualization tool application.

Following are the screenshots of the NeRFDeltaView Ensemble visualization tool application:
![NeRFDeltaView_Ensemble_A](https://github.com/CTW121/NeRFDeltaView-Ensemble/blob/master/images/NeRFDeltaView__Ensemble_A.png)

![NeRFDeltaView_Ensemble_B](https://github.com/CTW121/NeRFDeltaView-Ensemble/blob/master/images/NeRFDeltaView__Ensemble_B.png)

The demo videos can be found in [demo_video](https://github.com/CTW121/NeRFDeltaView-Ensemble/tree/master/demo_video).

## Color and Density Uncertainties Estimation
The color and density uncertainties are represented by the mean of the pairwise Euclidean distances because it considers a measure of variability between sample points within the color space. Figure below illustrates computation of color uncertainty $\delta_{\boldsymbol{c}}$ and density uncertainty $\delta_\sigma$ 3D grids. Read [master's thesis](https://github.com/CTW121/NeRFDeltaView-Ensemble/blob/master/Uncertainty_Visualization_for_Neural_Radiance_Field__Chua_T.W.pdf) for more detail.
![3D grid for color and density uncertainties](https://github.com/CTW121/NeRFDeltaView-Ensemble/blob/master/images/Ensemble_3D_regular_grids_color_density_uncertainties.png)

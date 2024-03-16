import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import copy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

class HeatMap(QVBoxLayout):
    def __init__(self, widget, data, vmin, vmax, data_angles, title, color, file_name):
        """
        Initialize a HeatMap object.

        Parameters:
        -----------
        widget: QWidget
            The parent widget to which the heatmap will be added.
        data: array-like
            The heatmap data.
        vmin: float
            Minimum value of the heatmap.
        vmax: float
            Maximum value of the heatmap.
        data_angles: array-like
            Angles of the data.
        title: str
            Title of the heatmap.
        color: str
            Color of the heatmap.
        file_name: str
            Name of the file to save the heatmap.

        Attributes:
        -----------
        data: array-like
            The heatmap data.
        vmin: float
            Minimum value of the heatmap.
        vmax: float
            Maximum value of the heatmap.
        data_angles: array-like
            Angles of the data.
        color: str
            Color of the heatmap.
        file_name: str
            Name of the file to save the heatmap.
        heatmap_selected_square: None
            Selected square on the heatmap.
        azimuth: list
            Azimuth angles for the heatmap.
        elevation: list
            Elevation angles for the heatmap.
        azimuth_label: list
            Azimuth angle labels for the heatmap.
        elevation_label: list
            Elevation angle labels for the heatmap.
        _main: QVBoxLayout
            Main layout for the heatmap widget.
        fig: Figure
            Figure for the heatmap.
        ax: Axes
            Axes for the heatmap.
        num_azimuth: int
            Number of azimuth angles.
        num_elevation: int
            Number of elevation angles.
        rectangles: list
            List to store rectangles for the heatmap.
        """
        super().__init__()

        # self.data = np.transpose(data)
        self.data = data
        self.vmin = vmin
        self.vmax = vmax
        self.data_angles = data_angles
        self.color = color
        self.file_name = file_name

        self.heatmap_selected_square = None

        # azimuth = [i for i in range(0, 360+1, 15)]
        # elevation = [i for i in range(0, 360+1, 15)]
        azimuth = [i for i in range(-180, 180+1, 15)]
        elevation = [i for i in range(-180, 180+1, 15)]


        # azimuth_label = [i for i in range(0, 360+1, 30)]
        # elevation_label = [i for i in range(0, 360+1, 30)]
        azimuth_label = [i for i in range(-180, 180+1, 30)]
        elevation_label = [i for i in range(-180, 180+1, 30)]

        self._main = QVBoxLayout(widget)

        # canvas = FigureCanvas(Figure(figsize=(10,10), dpi=100))
        canvas = FigureCanvas(Figure(figsize=(10,10)))
        self._main.addWidget(canvas)

        self.fig = canvas.figure
        self.ax = self.fig.subplots()
        # self.axes = copy.deepcopy(self.ax)  # Store the axes object
        self.ax.set_title(title, fontsize=11)
        self.ax.set_xlabel('Elevation ($\phi$)', fontsize=11)
        self.ax.set_ylabel('Azimuth ($\Theta$)', fontsize=11)

        self.ax.yaxis.set_label_coords(-0.1, 0.5)

        # # Set the limits of the scatter plot's axes based on the data dimensions
        self.num_azimuth = len(azimuth)
        self.num_elevation = len(elevation)
        # self.ax.set_xlim(0, self.num_elevation - 1)
        # self.ax.set_ylim(self.num_azimuth - 1, 0)
        self.ax.set_xlim(-0.5, self.num_elevation-0.5)
        self.ax.set_ylim(self.num_azimuth-0.5, -0.5)

        # self.ax.set_xticks(np.arange(len(azimuth)))
        self.ax.set_xticks([azimuth.index(value) for value in azimuth_label])
        self.ax.set_xticklabels(azimuth_label)

        # self.ax.set_yticks(np.arange(len(elevation)))
        self.ax.set_yticks([elevation.index(value) for value in elevation_label])
        self.ax.set_yticklabels(elevation_label)

        self.rectangles = []  # List to store rectangles

        self.create_heatmap()
        self.create_scatter_plot()
        # self.create_selection_square(12, 12, self.ax)

        # Save the heatmap as a png
        # figure_name = "{}.png".format(self.file_name)
        # self.fig.savefig(figure_name, format='png')
    
    def create_heatmap(self):
        """
        Create the heatmap visualization on the axes.

        This method adds the heatmap visualization to the axes, along with colorbar and annotations.

        Returns:
        --------
        None
        """
        self.heatmap = self.ax.imshow(self.data, cmap=self.color, interpolation='nearest', vmin=self.vmin, vmax=self.vmax)
        self.ax.invert_yaxis()
        cbar_heatmap = self.fig.colorbar(self.heatmap, fraction=0.046, pad=0.04)
        # cbar_heatmap.ax.set_yticklabels(['0.0', '0.25', '0.5'])
        cbar_heatmap.ax.tick_params(labelsize=7.5)

        # Add rectangles
        rectangles = [
                    Rectangle((6-0.4, 6-0.5), 13-0.2, 13, linewidth=3, edgecolor='blue', facecolor='none'),
                    Rectangle((0-0.4, 6-0.5), 6-0.2, 13, linewidth=3, edgecolor='green', facecolor='none'),
                    Rectangle((19-0.4, 6-0.5), 6-0.2, 13, linewidth=3, edgecolor='green', facecolor='none'),
                    ]
        
        for rect in rectangles:
            self.ax.add_patch(rect)
        
        # Add text for blue-bordered rectangle (upper hemisphere) and green-bordered rectangle (lower hemisphere)
        self.ax.text(0.5, 2.0, 'Blue rectangle: upper hemisphere', color='blue', fontsize=11)
        self.ax.text(0.5, 0.5, 'Green rectangle: lower hemisphere', color='green', fontsize=11)
    
    def create_scatter_plot(self):
        """
        Create the scatter plot visualization on the axes.

        This method scales the scatter plot data to match the dimensions of the heatmap and adds it to the axes.

        Returns:
        --------
        None
        """
        # Scale the scatter plot data to match the dimensions of the heatmap
        scaled_data_angles = np.zeros((len(self.data_angles),2))
        for i in range(len(self.data_angles)):
            scaled_data_angles[i, 0] = (self.data_angles[i,0]/360)*(self.num_elevation-1)
            scaled_data_angles[i, 1] = (self.data_angles[i,1]/360)*(self.num_azimuth-1)

        self.scatter_plot_angles = self.ax.scatter(scaled_data_angles[:,0], scaled_data_angles[:,1], color='blue', s=20, edgecolor='none', alpha=1.0)

    def selection_square(self, x, y, axis):
        """
        Create a selection square around a specified point on the heatmap.

        Parameters:
        -----------
        x : int
            The x-coordinate of the selected point.
        y : int
            The y-coordinate of the selected point.
        axis : matplotlib.axes.Axes
            The axes on which to draw the selection square.

        Returns:
        --------
        None
        """
        figure = self.ax.figure
        canvas = figure.canvas

        if len(self.rectangles)>0:
            # Remove rectangles
            for rect in self.rectangles:
                rect.remove()
            self.rectangles = []
        else:

            list_xy = [[x, y]]

            for i in range(len(list_xy)):
                if (list_xy[i][0]>=0 and list_xy[i][1]>=0) or (list_xy[i][0]<=24 and list_xy[i][1]<=24):
                    rect = Rectangle((list_xy[i][0] - 0.5, list_xy[i][1] - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=2)
                    self.ax.add_patch(rect)
                    self.rectangles.append(rect)

        canvas.draw_idle()
        canvas.flush_events()
        canvas.draw()
        canvas.flush_events()
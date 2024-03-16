import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.figure import Figure

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# from scipy.stats import kde
from scipy import stats

class ScatterPlot(QWidget):
    """
    A PyQt6 widget for displaying a scatter plot with a density plot overlay
    and supporting interactive selection of data points using a lasso selector tool.
    """

    def __init__(self, widget, selectInd, data):
        """
        Initializes the ScatterPlot widget.

        Parameters:
            widget (QWidget): The parent widget.
            selectInd (function): A function for handling selected indices.
            data (numpy.ndarray): The input data for the scatter plot.
        """
        super().__init__()

        # self.data = np.random.rand(10, 2)
        self.data = data
        self._main = QVBoxLayout(widget)

        canvas = FigureCanvas(Figure(figsize=(10,10)))
        self._main.addWidget(canvas)

        subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
        self.fig = canvas.figure
        self.ax = self.fig.subplots(subplot_kw=subplot_kw)
        # self.heatmap_ax = self.fig.subplots()
        self.ax.set_title("Density versus color uncertainties")
        self.ax.set_xlabel("Color uncertainty")
        self.ax.set_ylabel("Density uncertainty")

        self.selectInd = selectInd

        self.create_density_plot()
        self.create_scatterplot()

    def create_scatterplot(self):
        """
        Creates the scatter plot.
        """
        self.selected_ind = []
        # self.scatter_pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], color='red', s=20, edgecolor='none')
        self.scatter_pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], color='magenta', s=20, edgecolor='none')
        
    def create_density_plot(self):
        """
        Creates the density plot.
        """
        nbins = 80
        # data = self.data
        # k = stats.gaussian_kde([data[:, 0], data[:, 1]])
        k = stats.gaussian_kde([self.data[:, 0], self.data[:, 1]])
        xi, yi = np.mgrid[0.000:1.000:nbins*1j, 0.000:1.000:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        # ax = self.heatmap_ax
        density_plot = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Blues)
        self.fig.colorbar(density_plot)

    def select_from_collection(self, ax, collection, alpha_other=0.0):
        """
        Enables selection of data points using a lasso selector tool.

        Parameters:
            ax: The axes object.
            collection: Collection of scatter points.
            alpha_other (float): The alpha value for unselected points.
        """
        figure = self.ax.figure
        canvas = figure.canvas
        alpha_other = alpha_other

        xys = collection.get_offsets()
        Npts = len(xys)

        # Ensure that we have separate colors for each object
        fc = collection.get_facecolors()
        if len(fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(fc) == 1:
            fc = np.tile(fc, (Npts, 1))

        fc[:, -1] = 0.1
        collection.set_facecolors(fc)
        
        lineprops = {'color': 'black', 'linewidth': 1, 'alpha': 0.8 }
        lasso = LassoSelector(ax, onselect=lambda verts: onselect(verts, xys, fc, alpha_other), props=lineprops, useblit=True)
        ind = []
        selected_pts = []

        def onselect(verts, xys, fc, alpha_other):
            path = Path(verts)
            nonlocal ind
            ind = np.nonzero(path.contains_points(xys))[0]
            
            if len(ind) == 0:
                fc[:, -1] = 0.1
            else:
                fc[:, -1] = alpha_other
                fc[ind, -1] = 1

            # selected_pts = ind
            # print("self.data[ind]: \n", self.data[ind])
            # print("self.data[ind][:,2]: \n", self.data[ind][:,2].astype(int))
            # self.selectInd(ind)
            self.selectInd(self.data[ind][:,2].astype(int))

            self.selected_ind = ind

            collection.set_facecolors(fc)
            canvas.draw_idle()
            canvas.flush_events()
            canvas.draw()
            canvas.flush_events()

            # return selected_pts

        def disconnect():
            lasso.disconnect_events()
            fc[:, -1] = 1
            collection.set_facecolors(fc)
            canvas.draw_idle()
            # fig.canvas.flush_events()
            canvas.draw()

        self.ind = ind

        return ind, disconnect
        # return selected_pts, disconnect
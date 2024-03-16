import vtk

from PyQt6.QtWidgets import (
    QWidget,
    QSizePolicy,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QSplitter,
    QLabel,
    QPushButton,
    QTabWidget,
    QGridLayout,
    QSlider,
    QDoubleSpinBox,
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QObject,
    QSize,
    QEvent,
)
from PyQt6.QtGui import (
    QPalette,
    QColor,
    QCursor,
    QPixmap,
)

# Hide the three dots in the splitter handle
splitter_style_sheet = """
    QSplitter::handle {
        background: none;
        border: none;
    }
"""

frame_style_sheet_border = """
    QFrame {
        border: 1px solid green;
        border-radius: 10px;
    }
"""

frame_style_sheet = """
    QFrame {
        border: none;
    }
"""

frame_dict = {}
splitter_dict = {}
layout_dict = {}

def create_frame(name, style_sheet=None, width=None, height=None):
    """
    Creates a QFrame object with optional style sheet, width, and height.

    Parameters:
        name (str): Name to identify the frame.
        style_sheet (str, optional): Custom style sheet for the frame. Defaults to None.
        width (int, optional): Width of the frame. Defaults to None.
        height (int, optional): Height of the frame. Defaults to None.
    """
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)

    if style_sheet:
        frame.setStyleSheet(style_sheet)
    
    if width:
        frame.setFixedWidth(width)

    if height:
        frame.setFixedHeight(height)
    
    frame_dict[name] = frame

def create_splitter(name, widget_names, stretch_factors=[], style_sheet=None, orientation='horizontal'):
    """
    Creates a QSplitter object with specified widgets, stretch factors, style sheet, and orientation.

    Parameters:
        name (str): Name to identify the splitter.
        widget_names (list): Names of the QFrames to add to the splitter.
        stretch_factors (list, optional): List of tuples specifying stretch factors for widgets. Defaults to [].
        style_sheet (str, optional): Custom style sheet for the splitter. Defaults to None.
        orientation (str, optional): Orientation of the splitter ('horizontal' or 'vertical'). Defaults to 'horizontal'.
    """
    splitter = QSplitter(Qt.Orientation.Horizontal)
    if orientation == 'vertical':
        splitter = QSplitter(Qt.Orientation.Vertical)
    
    for widget_name in widget_names:
        splitter.addWidget(frame_dict[widget_name])
    
    for strecth_factor in stretch_factors:
        splitter.setStretchFactor(strecth_factor[0], strecth_factor[1])
    
    if style_sheet:
        splitter.setStyleSheet(style_sheet)
    
    splitter_dict[name] = splitter

def create_layout(name, frame, splitter=None, layout_type='QH'):
    """
    Creates a layout for a QFrame with optional splitter.

    Parameters:
        name (str): Name to identify the layout.
        frame (QFrame): The QFrame to apply the layout to.
        splitter (QSplitter, optional): The QSplitter to add to the layout. Defaults to None.
        layout_type (str, optional): Type of layout ('QH' for QHBoxLayout, 'QV' for QVBoxLayout). Defaults to 'QH'.
    """
    if layout_type == 'QH':
        layout = QHBoxLayout(frame)
    else:
        layout = QVBoxLayout(frame)
    
    if splitter:
        layout.addWidget(splitter)
    
    layout_dict[name] = layout
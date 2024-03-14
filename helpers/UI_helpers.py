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
    if layout_type == 'QH':
        layout = QHBoxLayout(frame)
    else:
        layout = QVBoxLayout(frame)
    
    if splitter:
        layout.addWidget(splitter)
    
    layout_dict[name] = layout
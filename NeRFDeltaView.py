# coding=utf-8
"""
Link:
https://gitlab.kitware.com/vtk/vtk/blob/master/Wrapping/Python/vtkmodules/qt/QVTKRenderWindowInteractor.py


A simple VTK widget for PyQt or PySide.
See http://www.trolltech.com for Qt documentation,
http://www.riverbankcomputing.co.uk for PyQt, and
http://pyside.github.io for PySide.

This class is based on the vtkGenericRenderWindowInteractor and is
therefore fairly powerful.  It should also play nicely with the
vtk3DWidget code.

Created by Prabhu Ramachandran, May 2002
Based on David Gobbi's QVTKRenderWidget.py

Changes by Gerard Vermeulen Feb. 2003
 Win32 support.

Changes by Gerard Vermeulen, May 2003
 Bug fixes and better integration with the Qt framework.

Changes by Phil Thompson, Nov. 2006
 Ported to PyQt v4.
 Added support for wheel events.

Changes by Phil Thompson, Oct. 2007
 Bug fixes.

Changes by Phil Thompson, Mar. 2008
 Added cursor support.

Changes by Rodrigo Mologni, Sep. 2013 (Credit to Daniele Esposti)
 Bug fix to PySide: Converts PyCObject to void pointer.

Changes by Greg Schussman, Aug. 2014
 The keyPressEvent function now passes keysym instead of None.

Changes by Alex Tsui, Apr. 2015
 Port from PyQt4 to PyQt5.

Changes by Fabian Wenzel, Jan. 2016
 Support for Python3

Changes by Tobias Hänel, Sep. 2018
 Support for PySide2

Changes by Ruben de Bruin, Aug. 2019
 Fixes to the keyPressEvent function

Changes by Chen Jintao, Aug. 2021
 Support for PySide6

Changes by Eric Larson and Guillaume Favelier, Apr. 2022
 Support for PyQt6
"""

import os
from turtle import color
import vtk
import numpy as np
import random
from PIL import Image
import json
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
# import SimpleITK as sitk
import math
import copy

import vtk.util.numpy_support as numpy_support

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy import stats
from sklearn.preprocessing import normalize

import subprocess

from helpers import (helpers, UI_helpers, ScatterPlot, HeatMap)


# Check whether a specific PyQt implementation was chosen
try:
    import vtkmodules.qt
    PyQtImpl = vtkmodules.qt.PyQtImpl
except ImportError:
    pass

# Check whether a specific QVTKRenderWindowInteractor base
# class was chosen, can be set to "QGLWidget" in
# PyQt implementation version lower than Qt6,
# or "QOpenGLWidget" in Pyside6 and PyQt6
QVTKRWIBase = "QWidget"
try:
    import vtkmodules.qt
    QVTKRWIBase = vtkmodules.qt.QVTKRWIBase
except ImportError:
    pass

from vtkmodules.vtkRenderingCore import vtkRenderWindow
from vtkmodules.vtkRenderingUI import vtkGenericRenderWindowInteractor

if PyQtImpl is None:
    # Autodetect the PyQt implementation to use
    try:
        import PyQt6.QtCore
        PyQtImpl = "PyQt6"
        # import PySide6.QtCore
        # PyQtImpl = "PySide6"
    except ImportError:
        try:
            import PySide6.QtCore
            PyQtImpl = "PySide6"
            # import PyQt6.QtCore
            # PyQtImpl = "PyQt6"
        except ImportError:
            try:
                import PyQt5.QtCore
                PyQtImpl = "PyQt5"
            except ImportError:
                try:
                    import PySide2.QtCore
                    PyQtImpl = "PySide2"
                except ImportError:
                    try:
                        import PyQt4.QtCore
                        PyQtImpl = "PyQt4"
                    except ImportError:
                        try:
                            import PySide.QtCore
                            PyQtImpl = "PySide"
                        except ImportError:
                            raise ImportError("Cannot load either PyQt or PySide")

# Check the compatibility of PyQtImpl and QVTKRWIBase
if QVTKRWIBase != "QWidget":
    if PyQtImpl in ["PySide6", "PyQt6"] and QVTKRWIBase == "QOpenGLWidget":
        pass  # compatible
    elif PyQtImpl in ["PyQt5", "PySide2","PyQt4", "PySide"] and QVTKRWIBase == "QGLWidget":
        pass  # compatible
    else:
        raise ImportError("Cannot load " + QVTKRWIBase + " from " + PyQtImpl)

if PyQtImpl == "PySide6":
    if QVTKRWIBase == "QOpenGLWidget":
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from PySide6.QtWidgets import QWidget
    from PySide6.QtWidgets import QSizePolicy
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QMainWindow
    from PySide6.QtGui import QCursor
    from PySide6.QtCore import Qt
    from PySide6.QtCore import QTimer
    from PySide6.QtCore import QObject
    from PySide6.QtCore import QSize
    from PySide6.QtCore import QEvent
elif PyQtImpl == "PyQt6":
    if QVTKRWIBase == "QOpenGLWidget":
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
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
        QSpinBox,
        QRadioButton,
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
    
elif PyQtImpl == "PyQt5":
    if QVTKRWIBase == "QGLWidget":
        from PyQt5.QtOpenGL import QGLWidget
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtWidgets import QSizePolicy
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWidgets import QMainWindow
    from PyQt5.QtGui import QCursor
    from PyQt5.QtCore import Qt
    from PyQt5.QtCore import QTimer
    from PyQt5.QtCore import QObject
    from PyQt5.QtCore import QSize
    from PyQt5.QtCore import QEvent
elif PyQtImpl == "PySide2":
    if QVTKRWIBase == "QGLWidget":
        from PySide2.QtOpenGL import QGLWidget
    from PySide2.QtWidgets import QWidget
    from PySide2.QtWidgets import QSizePolicy
    from PySide2.QtWidgets import QApplication
    from PySide2.QtWidgets import QMainWindow
    from PySide2.QtGui import QCursor
    from PySide2.QtCore import Qt
    from PySide2.QtCore import QTimer
    from PySide2.QtCore import QObject
    from PySide2.QtCore import QSize
    from PySide2.QtCore import QEvent
elif PyQtImpl == "PyQt4":
    if QVTKRWIBase == "QGLWidget":
        from PyQt4.QtOpenGL import QGLWidget
    from PyQt4.QtGui import QWidget
    from PyQt4.QtGui import QSizePolicy
    from PyQt4.QtGui import QApplication
    from PyQt4.QtGui import QMainWindow
    from PyQt4.QtCore import Qt
    from PyQt4.QtCore import QTimer
    from PyQt4.QtCore import QObject
    from PyQt4.QtCore import QSize
    from PyQt4.QtCore import QEvent
elif PyQtImpl == "PySide":
    if QVTKRWIBase == "QGLWidget":
        from PySide.QtOpenGL import QGLWidget
    from PySide.QtGui import QWidget
    from PySide.QtGui import QSizePolicy
    from PySide.QtGui import QApplication
    from PySide.QtGui import QMainWindow
    from PySide.QtCore import Qt
    from PySide.QtCore import QTimer
    from PySide.QtCore import QObject
    from PySide.QtCore import QSize
    from PySide.QtCore import QEvent
else:
    raise ImportError("Unknown PyQt implementation " + repr(PyQtImpl))

# Define types for base class, based on string
if QVTKRWIBase == "QWidget":
    QVTKRWIBaseClass = QWidget
elif QVTKRWIBase == "QGLWidget":
    QVTKRWIBaseClass = QGLWidget
elif QVTKRWIBase == "QOpenGLWidget":
    QVTKRWIBaseClass = QOpenGLWidget
else:
    raise ImportError("Unknown base class for QVTKRenderWindowInteractor " + QVTKRWIBase)

if PyQtImpl == 'PyQt6':
    CursorShape = Qt.CursorShape
    WidgetAttribute = Qt.WidgetAttribute
    FocusPolicy = Qt.FocusPolicy
    ConnectionType = Qt.ConnectionType
    Key = Qt.Key
    SizePolicy = QSizePolicy.Policy
    EventType = QEvent.Type
    try:
        MouseButton = Qt.MouseButton
        WindowType = Qt.WindowType
        KeyboardModifier = Qt.KeyboardModifier
    except AttributeError:
        # Fallback solution for PyQt6 versions < 6.1.0
        MouseButton = Qt.MouseButtons
        WindowType = Qt.WindowFlags
        KeyboardModifier = Qt.KeyboardModifiers
else:
    CursorShape = MouseButton = WindowType = WidgetAttribute = \
        KeyboardModifier = FocusPolicy = ConnectionType = Key = Qt
    SizePolicy = QSizePolicy
    EventType = QEvent

if PyQtImpl in ('PyQt4', 'PySide'):
    MiddleButton = MouseButton.MidButton
else:
    MiddleButton = MouseButton.MiddleButton


def _get_event_pos(ev):
    try:  # Qt6+
        return ev.position().x(), ev.position().y()
    except AttributeError:  # Qt5
        return ev.x(), ev.y()


class QVTKRenderWindowInteractor(QVTKRWIBaseClass):

    """ A QVTKRenderWindowInteractor for Python and Qt.  Uses a
    vtkGenericRenderWindowInteractor to handle the interactions.  Use
    GetRenderWindow() to get the vtkRenderWindow.  Create with the
    keyword stereo=1 in order to generate a stereo-capable window.

    The user interface is summarized in vtkInteractorStyle.h:

    - Keypress j / Keypress t: toggle between joystick (position
    sensitive) and trackball (motion sensitive) styles. In joystick
    style, motion occurs continuously as long as a mouse button is
    pressed. In trackball style, motion occurs when the mouse button
    is pressed and the mouse pointer moves.

    - Keypress c / Keypress o: toggle between camera and object
    (actor) modes. In camera mode, mouse events affect the camera
    position and focal point. In object mode, mouse events affect
    the actor that is under the mouse pointer.

    - Button 1: rotate the camera around its focal point (if camera
    mode) or rotate the actor around its origin (if actor mode). The
    rotation is in the direction defined from the center of the
    renderer's viewport towards the mouse position. In joystick mode,
    the magnitude of the rotation is determined by the distance the
    mouse is from the center of the render window.

    - Button 2: pan the camera (if camera mode) or translate the actor
    (if object mode). In joystick mode, the direction of pan or
    translation is from the center of the viewport towards the mouse
    position. In trackball mode, the direction of motion is the
    direction the mouse moves. (Note: with 2-button mice, pan is
    defined as <Shift>-Button 1.)

    - Button 3: zoom the camera (if camera mode) or scale the actor
    (if object mode). Zoom in/increase scale if the mouse position is
    in the top half of the viewport; zoom out/decrease scale if the
    mouse position is in the bottom half. In joystick mode, the amount
    of zoom is controlled by the distance of the mouse pointer from
    the horizontal centerline of the window.

    - Keypress 3: toggle the render window into and out of stereo
    mode.  By default, red-blue stereo pairs are created. Some systems
    support Crystal Eyes LCD stereo glasses; you have to invoke
    SetStereoTypeToCrystalEyes() on the rendering window.  Note: to
    use stereo you also need to pass a stereo=1 keyword argument to
    the constructor.

    - Keypress e: exit the application.

    - Keypress f: fly to the picked point

    - Keypress p: perform a pick operation. The render window interactor
    has an internal instance of vtkCellPicker that it uses to pick.

    - Keypress r: reset the camera view along the current view
    direction. Centers the actors and moves the camera so that all actors
    are visible.

    - Keypress s: modify the representation of all actors so that they
    are surfaces.

    - Keypress u: invoke the user-defined function. Typically, this
    keypress will bring up an interactor that you can type commands in.

    - Keypress w: modify the representation of all actors so that they
    are wireframe.
    """

    # Map between VTK and Qt cursors.
    _CURSOR_MAP = {
        0:  CursorShape.ArrowCursor,          # VTK_CURSOR_DEFAULT
        1:  CursorShape.ArrowCursor,          # VTK_CURSOR_ARROW
        2:  CursorShape.SizeBDiagCursor,      # VTK_CURSOR_SIZENE
        3:  CursorShape.SizeFDiagCursor,      # VTK_CURSOR_SIZENWSE
        4:  CursorShape.SizeBDiagCursor,      # VTK_CURSOR_SIZESW
        5:  CursorShape.SizeFDiagCursor,      # VTK_CURSOR_SIZESE
        6:  CursorShape.SizeVerCursor,        # VTK_CURSOR_SIZENS
        7:  CursorShape.SizeHorCursor,        # VTK_CURSOR_SIZEWE
        8:  CursorShape.SizeAllCursor,        # VTK_CURSOR_SIZEALL
        9:  CursorShape.PointingHandCursor,   # VTK_CURSOR_HAND
        10: CursorShape.CrossCursor,          # VTK_CURSOR_CROSSHAIR
    }

    def __init__(self, parent=None, **kw):
        # the current button
        self._ActiveButton = MouseButton.NoButton

        # private attributes
        self.__saveX = 0
        self.__saveY = 0
        self.__saveModifiers = KeyboardModifier.NoModifier
        self.__saveButtons = MouseButton.NoButton
        self.__wheelDelta = 0

        # do special handling of some keywords:
        # stereo, rw

        try:
            stereo = bool(kw['stereo'])
        except KeyError:
            stereo = False

        try:
            rw = kw['rw']
        except KeyError:
            rw = None

        # create base qt-level widget
        if QVTKRWIBase == "QWidget":
            if "wflags" in kw:
                wflags = kw['wflags']
            else:
                wflags = WindowType.Widget  # what Qt.WindowFlags() returns (0)
            QWidget.__init__(self, parent, wflags | WindowType.MSWindowsOwnDC)
        elif QVTKRWIBase == "QGLWidget":
            QGLWidget.__init__(self, parent)
        elif QVTKRWIBase == "QOpenGLWidget":
            QOpenGLWidget.__init__(self, parent)

        if rw: # user-supplied render window
            self._RenderWindow = rw
        else:
            self._RenderWindow = vtkRenderWindow()

        WId = self.winId()

        if type(WId).__name__ == 'PyCapsule':
            from ctypes import pythonapi, c_void_p, py_object, c_char_p

            pythonapi.PyCapsule_GetName.restype = c_char_p
            pythonapi.PyCapsule_GetName.argtypes = [py_object]

            name = pythonapi.PyCapsule_GetName(WId)

            pythonapi.PyCapsule_GetPointer.restype  = c_void_p
            pythonapi.PyCapsule_GetPointer.argtypes = [py_object, c_char_p]

            WId = pythonapi.PyCapsule_GetPointer(WId, name)

        self._RenderWindow.SetWindowInfo(str(int(WId)))

        if stereo: # stereo mode
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        try:
            self._Iren = kw['iren']
        except KeyError:
            self._Iren = vtkGenericRenderWindowInteractor()
            self._Iren.SetRenderWindow(self._RenderWindow)

        # do all the necessary qt setup
        self.setAttribute(WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(WidgetAttribute.WA_PaintOnScreen)
        self.setMouseTracking(True) # get all mouse events
        self.setFocusPolicy(FocusPolicy.WheelFocus)
        self.setSizePolicy(QSizePolicy(SizePolicy.Expanding, SizePolicy.Expanding))

        self._Timer = QTimer(self)
        self._Timer.timeout.connect(self.TimerEvent)

        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self._Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

        # If we've a parent, it does not close the child when closed.
        # Connect the parent's destroyed signal to this widget's close
        # slot for proper cleanup of VTK objects.
        if self.parent():
            self.parent().destroyed.connect(self.close, ConnectionType.DirectConnection)

    def __getattr__(self, attr):
        """Makes the object behave like a vtkGenericRenderWindowInteractor"""
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
        else:
            raise AttributeError(self.__class__.__name__ +
                  " has no attribute named " + attr)

    def Finalize(self):
        '''
        Call internal cleanup method on VTK objects
        '''
        self._RenderWindow.Finalize()

    def CreateTimer(self, obj, evt):
        self._Timer.start(10)

    def DestroyTimer(self, obj, evt):
        self._Timer.stop()
        return 1

    def TimerEvent(self):
        self._Iren.TimerEvent()

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render window."""
        # This indirection is needed since when the event fires, the current
        # cursor is not yet set so we defer this by which time the current
        # cursor should have been set.
        QTimer.singleShot(0, self.ShowCursor)

    def HideCursor(self):
        """Hides the cursor."""
        self.setCursor(CursorShape.BlankCursor)

    def ShowCursor(self):
        """Shows the cursor."""
        vtk_cursor = self._Iren.GetRenderWindow().GetCurrentCursor()
        qt_cursor = self._CURSOR_MAP.get(vtk_cursor, CursorShape.ArrowCursor)
        self.setCursor(qt_cursor)

    def closeEvent(self, evt):
        self.Finalize()

    def sizeHint(self):
        return QSize(400, 400)

    def paintEngine(self):
        return None

    def paintEvent(self, ev):
        self._Iren.Render()

    def resizeEvent(self, ev):
        scale = self._getPixelRatio()
        w = int(round(scale*self.width()))
        h = int(round(scale*self.height()))
        self._RenderWindow.SetDPI(int(round(72*scale)))
        vtkRenderWindow.SetSize(self._RenderWindow, w, h)
        self._Iren.SetSize(w, h)
        self._Iren.ConfigureEvent()
        self.update()

    def _GetKeyCharAndKeySym(self, ev):
        """ Convert a Qt key into a char and a vtk keysym.

        This is essentially copied from the c++ implementation in
        GUISupport/Qt/QVTKInteractorAdapter.cxx.
        """
        # if there is a char, convert its ASCII code to a VTK keysym
        try:
            keyChar = ev.text()[0]
            keySym = _keysyms_for_ascii[ord(keyChar)]
        except IndexError:
            keyChar = '\0'
            keySym = None

        # next, try converting Qt key code to a VTK keysym
        if keySym is None:
            try:
                keySym = _keysyms[ev.key()]
            except KeyError:
                keySym = None

        # use "None" as a fallback
        if keySym is None:
            keySym = "None"

        return keyChar, keySym

    def _GetCtrlShift(self, ev):
        ctrl = shift = False

        if hasattr(ev, 'modifiers'):
            if ev.modifiers() & KeyboardModifier.ShiftModifier:
                shift = True
            if ev.modifiers() & KeyboardModifier.ControlModifier:
                ctrl = True
        else:
            if self.__saveModifiers & KeyboardModifier.ShiftModifier:
                shift = True
            if self.__saveModifiers & KeyboardModifier.ControlModifier:
                ctrl = True

        return ctrl, shift

    @staticmethod
    def _getPixelRatio():
        if PyQtImpl in ["PyQt5", "PySide2", "PySide6", "PyQt6"]:
            # Source: https://stackoverflow.com/a/40053864/3388962
            pos = QCursor.pos()
            for screen in QApplication.screens():
                rect = screen.geometry()
                if rect.contains(pos):
                    return screen.devicePixelRatio()
            # Should never happen, but try to find a good fallback.
            return QApplication.instance().devicePixelRatio()
        else:
            # Qt4 seems not to provide any cross-platform means to get the
            # pixel ratio.
            return 1.

    def _setEventInformation(self, x, y, ctrl, shift,
                             key, repeat=0, keysum=None):
        scale = self._getPixelRatio()
        self._Iren.SetEventInformation(int(round(x*scale)),
                                       int(round((self.height()-y-1)*scale)),
                                       ctrl, shift, key, repeat, keysum)

    def enterEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self._setEventInformation(self.__saveX, self.__saveY,
                                  ctrl, shift, chr(0), 0, None)
        self._Iren.EnterEvent()

    def leaveEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        self._setEventInformation(self.__saveX, self.__saveY,
                                  ctrl, shift, chr(0), 0, None)
        self._Iren.LeaveEvent()

    def mousePressEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        repeat = 0
        if ev.type() == EventType.MouseButtonDblClick:
            repeat = 1
        x, y = _get_event_pos(ev)
        self._setEventInformation(x, y,
                                  ctrl, shift, chr(0), repeat, None)

        self._ActiveButton = ev.button()

        if self._ActiveButton == MouseButton.LeftButton:
            self._Iren.LeftButtonPressEvent()
        elif self._ActiveButton == MouseButton.RightButton:
            self._Iren.RightButtonPressEvent()
        elif self._ActiveButton == MiddleButton:
            self._Iren.MiddleButtonPressEvent()

    def mouseReleaseEvent(self, ev):
        ctrl, shift = self._GetCtrlShift(ev)
        x, y = _get_event_pos(ev)
        self._setEventInformation(x, y,
                                  ctrl, shift, chr(0), 0, None)

        if self._ActiveButton == MouseButton.LeftButton:
            self._Iren.LeftButtonReleaseEvent()
        elif self._ActiveButton == MouseButton.RightButton:
            self._Iren.RightButtonReleaseEvent()
        elif self._ActiveButton == MiddleButton:
            self._Iren.MiddleButtonReleaseEvent()

    def mouseMoveEvent(self, ev):
        self.__saveModifiers = ev.modifiers()
        self.__saveButtons = ev.buttons()
        x, y = _get_event_pos(ev)
        self.__saveX = x
        self.__saveY = y

        ctrl, shift = self._GetCtrlShift(ev)
        self._setEventInformation(x, y,
                                  ctrl, shift, chr(0), 0, None)
        self._Iren.MouseMoveEvent()

    def keyPressEvent(self, ev):
        key, keySym = self._GetKeyCharAndKeySym(ev)
        ctrl, shift = self._GetCtrlShift(ev)
        self._setEventInformation(self.__saveX, self.__saveY,
                                  ctrl, shift, key, 0, keySym)
        self._Iren.KeyPressEvent()
        self._Iren.CharEvent()

    def keyReleaseEvent(self, ev):
        key, keySym = self._GetKeyCharAndKeySym(ev)
        ctrl, shift = self._GetCtrlShift(ev)
        self._setEventInformation(self.__saveX, self.__saveY,
                                  ctrl, shift, key, 0, keySym)
        self._Iren.KeyReleaseEvent()

    def wheelEvent(self, ev):
        if hasattr(ev, 'delta'):
            self.__wheelDelta += ev.delta()
        else:
            self.__wheelDelta += ev.angleDelta().y()

        if self.__wheelDelta >= 120:
            self._Iren.MouseWheelForwardEvent()
            self.__wheelDelta = 0
        elif self.__wheelDelta <= -120:
            self._Iren.MouseWheelBackwardEvent()
            self.__wheelDelta = 0

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        self.update()


interactor_dict = dict()
scatter_plot_dict = dict()
histogram_dict = dict()
chart_dict = dict()
isosurface_dict = dict()
view_dict = dict()
heatmap_dict = dict()
rectangle_dict = dict()

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


def NeRFDeltaView():
    """NeRFDeltaView: Uncertainty visualization for Neural Radiance Fields"""
    
    colors = vtk.vtkNamedColors()

    # every QT app needs an app
    app = QApplication(['NeRFDeltaView • 神经辐射场不确定性可视化'])
    # Set the application-wide style to dark mode
    # app.setStyleSheet("QMainWindow { background-color: #333; color: white; }")

    window = QMainWindow()
    # window.setGeometry(0, 0, 1920, 1080)
    # window.setGeometry(0, 0, 1600, 900)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)

    layout = QHBoxLayout(central_widget)


    # Left frame
    UI_helpers.create_frame('frame_left', frame_style_sheet_border)
    UI_helpers.create_layout('layout_frame_left', frame=UI_helpers.frame_dict['frame_left'], layout_type='QV')


    # Middle frame
    UI_helpers.create_frame('frame_middle', frame_style_sheet_border)
    layout_frame_middle = QGridLayout(UI_helpers.frame_dict['frame_middle'])

    UI_helpers.create_frame('frame_middle_00', frame_style_sheet, width=500, height=500)
    UI_helpers.create_frame('frame_middle_01', frame_style_sheet, width=500, height=500)
    UI_helpers.create_frame('frame_middle_10', frame_style_sheet, width=500, height=500)
    UI_helpers.create_frame('frame_middle_11', frame_style_sheet, width=500, height=500)

    layout_frame_middle.addWidget(UI_helpers.frame_dict['frame_middle_00'], 0, 0)
    layout_frame_middle.addWidget(UI_helpers.frame_dict['frame_middle_01'], 0, 1)
    layout_frame_middle.addWidget(UI_helpers.frame_dict['frame_middle_10'], 1, 0)
    layout_frame_middle.addWidget(UI_helpers.frame_dict['frame_middle_11'], 1, 1)


    # Right frame
    UI_helpers.create_frame('frame_right', frame_style_sheet_border)
    UI_helpers.create_layout('layout_frame_right', UI_helpers.frame_dict['frame_right'])

    tab_widget = QTabWidget()
    tab_widget.setStyleSheet(frame_style_sheet)
    tab_widget.tabBar().setStyleSheet("color: green")
    # tab_widget.tabBar().setStyleSheet("color: white")
    # tab_widget.setStyleSheet("QTabWidget::pane { background-color: #444; color: black; }"
    #                      "QTabWidget::tab-bar { alignment: left; }"
    #                      "QTabBar::tab { background-color: #444; color: white; }"
    #                      "QTabBar::tab:selected { background-color: #444; }"
    #                      "QFrame { border: none; }")

    tab1_1DTF = QWidget()
    # tab1_1DTF.setAutoFillBackground(True)
    # tab1_palette = tab1_1DTF.palette()
    # tab1_palette.setColor(QPalette.ColorRole.Window, QColor(44, 44, 44))  # Dark gray background
    # tab1_1DTF.setPalette(tab1_palette)

    tab2_2DTF_mean_sd = QWidget()
    # tab2_2DTF_mean_sd.setAutoFillBackground(True)
    # tab2_palette = tab2_2DTF_mean_sd.palette()
    # tab2_palette.setColor(QPalette.ColorRole.Window, QColor(44, 44, 44))  # Dark gray background
    # tab2_2DTF_mean_sd.setPalette(tab1_palette)

    # tab_widget.addTab(tab1_1DTF, "1-D transfer functions and 2-D scatter plot")
    # tab_widget.addTab(tab2_2DTF_mean_sd, "mean and standard deviation")
    tab_widget.addTab(tab1_1DTF, "Transfer functions")
    tab_widget.addTab(tab2_2DTF_mean_sd, "Heatmaps")

    UI_helpers.layout_dict['layout_frame_right'].addWidget(tab_widget)
    UI_helpers.frame_dict['frame_right'].setLayout(UI_helpers.layout_dict['layout_frame_right'])


    # Main splitter
    UI_helpers.create_splitter('splitter_main', ['frame_left', 'frame_middle', 'frame_right'], [[0,1], [1,1], [2,1]], style_sheet=splitter_style_sheet, orientation='horizontal')

    layout.addWidget(UI_helpers.splitter_dict['splitter_main'])


    ### ==================================================== ###
    ### === VISUALIZATION (1D and 2D transfer functions) === ###
    ### ==================================================== ###

    # ========= Parameters ========= #
    dataset = "chair"            # dataset: lego / hotdog / chair
    dataset_size = "full"       # full or partial
    iterations = 200000         # number of iterations

    # isofurface value
    isosurface_filter_value = 0.90

    # mask
    color_uncertainty_filter = 0.01      # remove the scatter points of color uncertainty lower than this value
    density_uncertainty_filter = 0.01    # remove the scatter points of density uncertainty lower than this value
    histogram_color_uncertainty_filter = 0.01       # remove the bar in the color uncertainty histogram lower than this value
    histogram_density_uncertainty_filter = 0.01     # remove the bar in the density uncertainty histogram lower than this value

    # heatmap colorbar scale bound
    # color_mean_min = 
    # color_mean_max = 

    # color_stddev_min = 
    # color_stddev_max = 

    # density_mean_min = 
    # density_mean_max = 

    # density_stddev_min = 
    # density_stddev_max = 
    # ========= Parameters ========= #


    # folder "data" stores the VTK and CSV files
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


    # Define the file paths including the folder
    color_means_file = os.path.join(data_folder, "color_means.csv")
    color_stddev_file = os.path.join(data_folder, "color_standard_deviations.csv")
    
    density_means_file = os.path.join(data_folder, "density_means.csv")
    density_stddev_file = os.path.join(data_folder, "density_standard_deviations.csv")

    # color_means = loadtxt("color_means.csv", delimiter=",")
    # density_means = loadtxt("density_means.csv", delimiter=",")

    # color_standard_deviation = loadtxt("color_standard_deviations.csv", delimiter=",")
    # density_standard_deviation = loadtxt("density_standard_deviations.csv", delimiter=",")

    color_means = loadtxt(color_means_file, delimiter=",")
    density_means = loadtxt(density_means_file, delimiter=",")

    color_standard_deviation = loadtxt(color_stddev_file, delimiter=",")
    density_standard_deviation = loadtxt(density_stddev_file, delimiter=",")

    angles_file = os.path.join(data_folder, "angles_{}__{}.csv".format(dataset_size, dataset))
    angles = loadtxt(angles_file, delimiter=",")

    print("min(color_means): ", np.min(color_means))
    print("max(color_means): ", np.max(color_means))
    print()
    print("min(color_standard_deviation): ", np.min(color_standard_deviation))
    print("max(color_standard_deviation): ", np.max(color_standard_deviation))
    print()
    print("min(density_means): ", np.min(density_means))
    print("max(density_means): ", np.max(density_means))
    print()
    print("min(density_standard_deviation): ", np.min(density_standard_deviation))
    print("max(density_standard_deviation): ", np.max(density_standard_deviation))

    color_mean_min = np.min(color_means)
    color_mean_max = np.max(color_means)

    color_stddev_min = np.min(color_standard_deviation)
    color_stddev_max = np.max(color_standard_deviation)

    density_mean_min = np.min(density_means)
    density_mean_max = np.max(density_means)

    density_stddev_min = np.min(density_standard_deviation)
    density_stddev_max = np.max(density_standard_deviation)


    """
    Isosurface

    Set up the rendering of an isosurface within a designated frame
    """
    frame_middle_10 = UI_helpers.frame_dict['frame_middle_10']
    interactor_isosurface = QVTKRenderWindowInteractor(frame_middle_10)
    frame_middle_10.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_middle_10, interactor_isosurface)

    style_isosurface = helpers.vtk_create_interactor_style(interactor_isosurface)
    
    contour_opacity = helpers.vtk_contour_filter(opacity_volume, filter_value=isosurface_filter_value)
    mapper_opacity = helpers.vtk_poly_data_mapper(contour_opacity)
    actor_opacity = helpers.vtk_create_actor(mapper_opacity, 'Green')


    """
    Outline

    Generate an outline representation of the volume data for visualization purposes
    """
    outline = helpers.vtk_create_outline(opacity_reader)


    """
    Opacity (for visualizing the 3D reconstruction model)

    Set up the opacity visualization for the 3D reconstruction model
    """
    # the opacity volume is identical for color uncertainty volume and density uncertainty volume
    frame_middle_01 = UI_helpers.frame_dict['frame_middle_01']
    frame_middle_10 = UI_helpers.frame_dict['frame_middle_11']
    interactor_opacity_colorUnc = QVTKRenderWindowInteractor(frame_middle_01)
    interactor_opacity_densityUnc = QVTKRenderWindowInteractor(frame_middle_10)
    frame_middle_01.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_middle_01, interactor_opacity_colorUnc)
    frame_middle_10.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_middle_10, interactor_opacity_densityUnc)

    style_colorUnc = helpers.vtk_create_interactor_style(interactor_opacity_colorUnc)
    style_densityUnc = helpers.vtk_create_interactor_style(interactor_opacity_densityUnc)

    opacity_tf_Opacity = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.60, 1.00], [1.00, 0.99]])
    alpha_tf_Opacity = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 0.0, 0.0, 0.0], [1.00, 1.0, 1.0, 1.0]])
    volume_property_Opacity = helpers.vtk_volume_property(alpha_tf_Opacity, opacity_tf_Opacity)
    volume_mapper_Opacity = helpers.vtk_volume_ray_cast_mapper(opacity_reader)
    volume_opacity = helpers.vtk_create_volume(volume_mapper_Opacity, volume_property_Opacity)


    """
    Color uncertainty

    Set up the visualization for representing color uncertainty
    """
    opacity_tf_color_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.85, 1.00]])
    alpha_tf_color_uncertainty = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 1.0, 1.0, 1.0], [1.00, 1.0, 0.0, 0.0]])
    volume_property_color_uncertainty = helpers.vtk_volume_property(alpha_tf_color_uncertainty, opacity_tf_color_uncertainty)
    volume_mapper_color_uncertainty = helpers.vtk_volume_ray_cast_mapper(color_uncertainty_reader)
    volume_color_uncertainty = helpers.vtk_create_volume(volume_mapper_color_uncertainty, volume_property_color_uncertainty)


    """
    Density uncertainty

    Configure the visualization of density uncertainty
    """
    opacity_tf_density_uncertainty = helpers.vtk_create_piecewise_function([[0.00, 0.00], [0.60, 1.00]])
    alpha_tf_density_uncertainty = helpers.vtk_create_color_transfer_function("RGB", [[0.00, 1.0, 1.0, 1.0], [1.00, 1.0, 0.647, 0.0]])
    volume_property_density_uncertainty = helpers.vtk_volume_property(alpha_tf_density_uncertainty, opacity_tf_density_uncertainty)
    volume_mapper_density_uncertainty = helpers.vtk_volume_ray_cast_mapper(density_uncertainty_reader)
    volume_density_uncertainty = helpers.vtk_create_volume(volume_mapper_density_uncertainty, volume_property_density_uncertainty)


    """
    Scalar bar (color bar)

    Create scalar bars to represent the color uncertainty and density uncertainty
    """
    scalar_bar_color_uncertainty, scalar_bar_color_uncertainty_widget = helpers.vtk_create_scalar_bar(alpha_tf_color_uncertainty, interactor_opacity_colorUnc, "$\delta_{color}$")
    scalar_bar_density_uncertainty, scalar_bar_density_uncertainty_widget = helpers.vtk_create_scalar_bar(alpha_tf_density_uncertainty, interactor_opacity_densityUnc, "$\delta_{density}$")


    """
    Title for each render window
    """
    title_isosurface = helpers.vtk_create_text_actor(interactor_isosurface, "Isosurface")
    title_colorUnc = helpers.vtk_create_text_actor(interactor_opacity_colorUnc, "Uncertainty: color")
    title_densityUnc = helpers.vtk_create_text_actor(interactor_opacity_densityUnc, "Uncertainty: density")


    """
    Renderers

    Renderer objects are created for different visualizations
    """
    renderer_isosurface = helpers.vtk_create_renderer(actor_opacity, title_isosurface)
    interactor_isosurface.GetRenderWindow().AddRenderer(renderer_isosurface)

    renderer_opacity_colorUnc = helpers.vtk_create_renderer(volume_opacity, volume_color_uncertainty, scalar_bar_color_uncertainty, title_colorUnc)
    interactor_opacity_colorUnc.GetRenderWindow().AddRenderer(renderer_opacity_colorUnc)

    renderer_opacity_densityUnc = helpers.vtk_create_renderer(volume_opacity, volume_density_uncertainty, scalar_bar_density_uncertainty, title_densityUnc)
    interactor_opacity_densityUnc.GetRenderWindow().AddRenderer(renderer_opacity_densityUnc)


    """
    Z-buffer (for uncertainties from iso-surface)

    vtkFloatArray objects are created to store depth buffer data for different visualizations
    """
    z_buffer_data_isosurface = vtk.vtkFloatArray()
    z_buffer_data_color_uncertainty = vtk.vtkFloatArray()
    z_buffer_data_density_uncertainty = vtk.vtkFloatArray()

    def zBuffer(obj, key):
        """
        Toggle Z-buffer preservation for rendering uncertainty visualizations alongside the isosurface.
        
        Parameters:
        -----------
        obj : vtk.vtkObject
            The VTK object associated with the event.
        key : str
            The event key.

        Return:
        -------
        None

        Notes:
        ------
        This function toggles Z-buffer preservation for the isosurface, color uncertainty, and density uncertainty renderers based on the global variable 'radio_button_opacity'. If 'radio_button_opacity' is True, indicating that the scene geometry is displayed, Z-buffer preservation is turned off for all renderers, and depth buffer data is retrieved and stored. If 'radio_button_opacity' is False, indicating that the scene geometry is hidden, Z-buffer preservation is turned off for all renderers. Finally, render updates are triggered for all interactor windows to reflect the changes.
        """
        # global variable from the radio button which display or remove the scene geometry
        # the radio button is located beneath the histrogram of the scene
        global radio_button_opacity

        if radio_button_opacity:
            renderer_isosurface.PreserveDepthBufferOff()
            renderer_isosurface.GetRenderWindow().Render()

            renderer_opacity_colorUnc.PreserveDepthBufferOff()
            renderer_opacity_colorUnc.GetRenderWindow().Render()

            renderer_opacity_densityUnc.PreserveDepthBufferOff()
            renderer_opacity_densityUnc.GetRenderWindow().Render()

            # renderer_isosurface.PreserveDepthBufferOn()
            xmax_isosurface, ymax_isosurface = renderer_isosurface.GetRenderWindow().GetActualSize()
            renderer_isosurface.GetRenderWindow().GetZbufferData(0, 0, ymax_isosurface-1, xmax_isosurface-1, z_buffer_data_isosurface)
            # for i in range(z_buffer_data_tr.GetNumberOfTuples()):
            #     current_depth = z_buffer_data_isosurface.GetValue(i)
            #     z_buffer_data_isosurface.SetValue(i, current_depth + 0.008)   # Z-buffer extra depth
            # z_buffer_data_isosurface_np = numpy_support.vtk_to_numpy(z_buffer_data_isosurface)
            # print("isosurface: ", np.unique(z_buffer_data_isosurface_np))

            # Print depth map
            # -------------------------------------------------------
            # z_buffer_data_np = numpy_support.vtk_to_numpy(z_buffer_data_isosurface)
            # depth_map_reshape = z_buffer_data_np.reshape(ymax_isosurface, xmax_isosurface)
            # depth_map_reshape_flip = np.flipud(depth_map_reshape)
            # fig, ax = plt.subplots()
            # ax.imshow(depth_map_reshape_flip, cmap='gray')
            # ax.axis('off')
            # plt.savefig('depth_map.png', format='png')
            # plt.close(fig)
            # -------------------------------------------------------

            xmax_colorUnc, ymax_colorUnc = renderer_opacity_colorUnc.GetRenderWindow().GetActualSize()
            renderer_opacity_colorUnc.PreserveDepthBufferOn()
            renderer_opacity_colorUnc.GetRenderWindow().GetZbufferData(0, 0, ymax_colorUnc-1, xmax_colorUnc-1, z_buffer_data_color_uncertainty)
            renderer_opacity_colorUnc.GetRenderWindow().SetZbufferData(0, 0, ymax_colorUnc-1, xmax_colorUnc-1, z_buffer_data_isosurface)

            xmax_densityUnc, ymax_densityUnc = renderer_opacity_densityUnc.GetRenderWindow().GetActualSize()
            renderer_opacity_densityUnc.PreserveDepthBufferOn()
            renderer_opacity_densityUnc.GetRenderWindow().GetZbufferData(0, 0, ymax_densityUnc-1, xmax_densityUnc-1, z_buffer_data_density_uncertainty)
            renderer_opacity_densityUnc.GetRenderWindow().SetZbufferData(0, 0, ymax_densityUnc-1, xmax_densityUnc-1, z_buffer_data_isosurface)

        else:
            renderer_isosurface.PreserveDepthBufferOff()
            renderer_isosurface.GetRenderWindow().Render()

            renderer_opacity_colorUnc.PreserveDepthBufferOff()
            renderer_opacity_colorUnc.GetRenderWindow().Render()

            renderer_opacity_densityUnc.PreserveDepthBufferOff()
            renderer_opacity_densityUnc.GetRenderWindow().Render()


        interactor_isosurface.GetRenderWindow().Render()
        interactor_opacity_colorUnc.GetRenderWindow().Render()
        interactor_opacity_densityUnc.GetRenderWindow().Render()
        
    
    interactor_isosurface.AddObserver('EndInteractionEvent', zBuffer)
    interactor_opacity_colorUnc.AddObserver('EndInteractionEvent', zBuffer)
    interactor_opacity_densityUnc.AddObserver('EndInteractionEvent', zBuffer)


    """
    camera for all render windows

    Configure the camera settings for all render windows.

    Parameters:
    -----------
    None

    Notes:
    ------
    This section configures the camera settings for all render windows to ensure consistent viewing angles and 
    zoom levels. The camera settings are adjusted to zoom in, dolly closer, and 
    rotate for a better perspective. 
    The same camera settings are applied to both the color uncertainty and 
    density uncertainty renderers to maintain consistency across visualizations.
    """
    camera = renderer_isosurface.GetActiveCamera()

    original_orient = helpers.vtk_get_orientation(renderer_isosurface)
    
    camera.Zoom(5.0)
    camera.Dolly(0.2)
    camera.Pitch(45)
    camera.Yaw(-80)

    renderer_opacity_colorUnc.SetActiveCamera(camera)
    renderer_opacity_colorUnc.ResetCamera()

    renderer_opacity_densityUnc.SetActiveCamera(camera)
    renderer_opacity_densityUnc.ResetCamera()


    """
    Plane Widget

    Configure a plane widget for interactive slicing.

    Parameters:
    -----------
    None

    Notes:
    ------
    This section sets up a plane widget for interactive slicing of the 3D visualization. 
    The plane widget is attached to the isosurface renderer's interactor for interactive manipulation. 
    The plane's position and orientation are updated based on user interaction, 
    triggering a refresh of the render windows to reflect the changes in slicing.
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(mapper_opacity.GetCenter())
    plane.SetNormal(1, 0, 0)

    mapper_opacity.AddClippingPlane(plane)
    volume_mapper_Opacity.AddClippingPlane(plane)
    volume_mapper_color_uncertainty.AddClippingPlane(plane)
    volume_mapper_density_uncertainty.AddClippingPlane(plane)

    def planeObserver(obj, event):
        plane.SetOrigin(obj.GetCenter())
        plane.SetNormal(obj.GetNormal())
        zBuffer(None,None)
        interactor_isosurface.GetRenderWindow().Render()
        interactor_opacity_colorUnc.GetRenderWindow().Render()
        interactor_opacity_densityUnc.GetRenderWindow().Render()

    planeWidget = vtk.vtkImagePlaneWidget()
    planeWidget.SetInteractor(interactor_isosurface)
    # planeWidget.SetInteractor(interactor_opacity_colorUnc)
    # planeWidget.SetInteractor(interactor_opacity_densityUnc)
    planeWidget.TextureVisibilityOff()
    planeWidget.UpdatePlacement()
    planeWidget.AddObserver('EndInteractionEvent', planeObserver)
    planeWidget.On()



    def onTransferFunctionPointModified(obj, ev):
        """
        Transfer Function Point Modification Handler

        Handle modification events of transfer function points.

        Parameters:
        -----------
        obj : vtkObject
            The object triggering the event.
        ev : str
            The event type.

        Returns:
        --------
        None

        Notes:
        ------
        This function is responsible for updating the scalar opacity properties of the visualization volumes 
        when transfer function points are modified. It sets the scalar opacity properties of the opacity volume 
        and the color uncertainty volume based on the modified transfer functions, 
        then triggers a render update for the color uncertainty and density uncertainty render windows.
        """
        volume_opacity.GetProperty().SetScalarOpacity(opacity_tf_Opacity)
        volume_color_uncertainty.GetProperty().SetScalarOpacity(opacity_tf_color_uncertainty)
        interactor_opacity_colorUnc.GetRenderWindow().Render()
        interactor_opacity_densityUnc.GetRenderWindow().Render()
    

    """
    Import rendered figure (generated by NeRF model using PyTorch)

    Import and display a rendered figure generated by a NeRF model using PyTorch.
    """
    frame_middle_00 = UI_helpers.frame_dict['frame_middle_00']
    frame_middle_00_geometry = frame_middle_00.frameRect()
    frame_middle_00_geometry_width = frame_middle_00_geometry.width()
    frame_middle_00_geometry_height = frame_middle_00_geometry.height()
    image_label = QLabel(frame_middle_00)
    pixmap = QPixmap("text_render_image.png")
    resized_pixmap = pixmap.scaled(frame_middle_00_geometry_width, frame_middle_00_geometry_height)
    image_label.setPixmap(resized_pixmap)

    frame_middle_00_boxlayout = QVBoxLayout(frame_middle_00)
    frame_middle_00_boxlayout.addWidget(image_label)
    frame_middle_00.setLayout(frame_middle_00_boxlayout)


    """
    Text
    Information of the 3D volume

    Display information about the 3D volume, including file properties and data statistics.

    Notes:
    ------
    This section creates QLabel widgets to display information about the 3D volume. 
    It includes details such as file properties (file names and iterations), 
    data type, dimensions, number of points, and bounds. 
    The information is formatted using HTML tags for styling and layout purposes.
    """
    label_layout = QVBoxLayout()

    path = os.path.abspath(opacity_file_name)
    dir = os.path.dirname(path)
    # label_file_text = "<h4 style='color:Green;'> File properties </h4> <p>Files: <br> {} <br> {} <br> {}</p> <p>Path: <br> {}</p> <br>".format(opacity_file_name, color_uncertainty_file_name, density_uncertainty_file_name, dir)
    label_file_text = "<h4 style='color:Green;'> File properties </h4> <p>Files: <br> {} <br> {} <br> {}</p> <p>Iterations: <br> {} </p> <br>".format(opacity_file_name, color_uncertainty_file_name, density_uncertainty_file_name, iterations)

    label_file = QLabel(label_file_text)
    label_file.setStyleSheet("border: 0px;")

    data_type_id = opacity_volume.GetDataObjectType()
    data_type_name = vtk.vtkDataObjectTypes.GetClassNameFromTypeId(data_type_id)
    # data_opacity_, dims_opacity_ = helpers.vtk_structured_point_value_array(opacity_reader)
    dims_opacity_ = opacity_reader.GetOutput().GetDimensions()
    num_points_opacity = dims_opacity_[0] * dims_opacity_[1] * dims_opacity_[2]

    xmin, xmax, ymin, ymax, zmin, zmax = opacity_volume.GetBounds()

    label_text = "<h4 style='color:Green;'>Data statistics</h4> <p>Type: <br> {}</p> <p>Dimensions: <br> x: {} <br> y: {} <br> z: {} </p> <p>Number of points: <br> {}</p> <p> Bounds: <br> x: {} to {} <br> y: {} to {} <br> z: {} to {} </p>" \
                .format(data_type_name, dims_opacity_[0], dims_opacity_[1], dims_opacity_[2], num_points_opacity, round(xmin,2), round(xmax,2), round(ymin,2), round(ymax,2), round(zmin,2), round(zmax,2))
    label = QLabel(label_text)
    # label.setStyleSheet("border: 0px; background: white; color: black;")
    label.setStyleSheet("border: 0px;")

    label_layout.addWidget(label_file)
    label_layout.addWidget(label)


    """
    Button (generated by QButton)
    For generating the volume rendering image generated by PyTorch model

    Create a button using QPushButton for generating the volume rendering image generated by the PyTorch model.

    Notes:
    ------
    This section creates a QPushButton labeled "Render Image" along with a QLabel providing context. 
    The QPushButton triggers the generation of a volume rendering image using the PyTorch model when clicked. 
    The layout is adjusted for proper alignment using QVBoxLayout with specified margins.
    """
    button_layout = QVBoxLayout()
    button_layout.setContentsMargins(0, 30, 0, 85)

    label_button_text = "<h4 style='color:Green;'> Create a rendered image using PyTorch <\h4>"
    label_button = QLabel(label_button_text)
    label_button.setStyleSheet("border: 0px;")

    button = QPushButton("Render Image")
    # button.setStyleSheet("border: 0px;")

    button_layout.addWidget(label_button)
    button_layout.addWidget(button)

    def show_image(self):
        import eval_nerf
        
        helpers.vtk_get_orientation(renderer_isosurface)
        # camera = renderer_isosurface.GetActiveCamera()
        matrix = camera.GetModelViewTransformMatrix()
        
        rotation_matrix = np.eye(4)
        for row in range(3):
            for col in range(3):
                rotation_matrix[row, col] = matrix.GetElement(row, col)

        # Get the camera position and focal point
        cameraPosition = camera.GetPosition()
        focalPoint = camera.GetFocalPoint()

        # Calculate the vector between camera and focal point
        vector = np.array([cameraPosition[i] - focalPoint[i] for i in range(3)])
        vector_magnitude = np.linalg.norm(vector)

        # rendered 3D scene using PyTorch
        focal_length = 1200

        eval_nerf.get_render_image(vector_magnitude, rotation_matrix, focal_length, iterations)

        # display the rendered image in the application's frame
        pixmap = QPixmap("0000.png")
        resized_pixmap = pixmap.scaled(frame_middle_00_geometry_width, frame_middle_00_geometry_height)
        image_label.setPixmap(resized_pixmap)
    
    button.clicked.connect(show_image)


    """
    Double spin box (generated by QDoubleSpinBox)
    For adjusting the isosurface value

    Create a double spin box using QDoubleSpinBox for adjusting the isosurface value

    Notes:
    ------
    This section creates a QDoubleSpinBox labeled "Isosurface value" along with a QLabel providing context. 
    The QDoubleSpinBox allows users to adjust the isosurface value with a single step size of 0.01 and 
    an initial value of 0.90. The layout is adjusted for proper alignment using QVBoxLayout with specified margins.
    """
    double_spin_box_layout = QVBoxLayout()
    double_spin_box_layout.setContentsMargins(0, 30, 0, 85)

    label_double_spin_box_text = "<h4 style='color:Green;'> Isosurface value <\h4>"
    label_double_spin_box = QLabel(label_double_spin_box_text)
    label_double_spin_box.setStyleSheet("border: 0px;")

    double_spin_box = QDoubleSpinBox()
    double_spin_box.setSingleStep(0.01)
    double_spin_box.setValue(0.90)

    double_spin_box_layout.addWidget(label_double_spin_box)
    double_spin_box_layout.addWidget(double_spin_box)


    """
    Spin box (generated by QSpinBox)
    For adjusting the number of bins of the histogram

    Create a spin box using QSpinBox for adjusting the number of bins in the histogram.

    Notes:
    ------
    This section creates a QSpinBox labeled "Number of bins in the histogram" along with a QLabel providing context. 
    The QSpinBox allows users to adjust the number of bins in the histogram with an initial value of 20. 
    The layout is adjusted for proper alignment using QVBoxLayout with specified margins.
    """
    spin_box_layout = QVBoxLayout()
    spin_box_layout.setContentsMargins(0, 30, 0, 85)

    label_spin_box_text = "<h4 style='color:Green;'> Number of bins in the histogram <\h4>"
    label_spin_box = QLabel(label_spin_box_text)
    label_spin_box.setStyleSheet("border: 0px;")

    spin_box = QSpinBox()
    spin_box.setValue(20)

    spin_box_layout.addWidget(label_spin_box)
    spin_box_layout.addWidget(spin_box)


    """
    Box
    For showing azimuth, elevation and the uncertainty value selected from
    color and density uncertainty heatmaps

    Create a box layout for showing azimuth, elevation, and the uncertainty value selected from color and density uncertainty heatmaps.

    Notes:
    ------
    This section defines a QVBoxLayout named heatmap_label_layout with specified margins for proper alignment. 
    It contains a QLabel displaying text about mean and standard deviation from the heatmap, 
    along with placeholders for elevation, azimuth, color mean, color standard deviation, density mean, 
    and density standard deviation. These values are updated dynamically based on user interactions.
    """
    heatmap_label_layout = QVBoxLayout()
    heatmap_label_layout.setContentsMargins(0, 30, 0, 5)

    # label_heatmap_text = "<style type='text/css'> h4 {color: green;} </style> <h4> Mean and standard deviation (SD) <br> from heatmap </h4> <p> Azimuth: <br> Elevation: <br> <br> Color mean: <br> Color SD: <br> <br> Density mean: <br> Density SD: </p>"
    label_heatmap_text = "<h4 style='color:Green;'> Mean and standard deviation (SD) <br> from heatmap </h4> <p> Elevation: <br> Azimuth: <br> <br> Color mean: <br> Color SD: <br> <br> Density mean: <br> Density SD: </p>"
    label_heatmap = QLabel(label_heatmap_text)
    label_heatmap.setStyleSheet("border: 0px;")
    heatmap_label_layout.addWidget(label_heatmap)

    layout_frame_left = UI_helpers.layout_dict['layout_frame_left']
    layout_frame_left.addLayout(label_layout)
    layout_frame_left.addLayout(button_layout)
    # layout_frame_left.addLayout(double_spin_box_layout)
    layout_frame_left.addLayout(spin_box_layout)
    # layout_frame_left.addLayout(heatmap_color_label_layout)
    # layout_frame_left.addLayout(heatmap_density_label_layout)
    layout_frame_left.addLayout(heatmap_label_layout)

    data_opacity = np.array(opacity_volume.GetPointData().GetScalars())
    data_color = np.array(color_uncertainty_volume.GetPointData().GetScalars())
    data_density = np.array(density_uncertainty_volume.GetPointData().GetScalars())
    
    def update_histogram(num_bins, selected_scatter_pts=False):
        """
        Update the histogram data based on the specified number of bins.

        Parameters:
        -----------
        num_bins : int
            Number of bins for the histogram.
        selected_scatter_pts : bool, optional
            Flag indicating whether selected scatter points are used.

        Returns:
        --------
        None

        Notes:
        ------
        This function computes normalized histograms for opacity, color, and 
        density data using the specified number of bins. 
        It then updates the histogram data in the transfer function for each type of data. 
        Finally, it redraws the histograms to reflect the changes.
        """
        hist_norm_opacity = helpers.create_histogram_array(data_opacity, num_bins=num_bins, filter=False)

        hist_norm_color = helpers.create_histogram_array(data_color, num_bins=num_bins, filter=True, filter_threshold=histogram_color_uncertainty_filter)

        hist_norm_density = helpers.create_histogram_array(data_density, num_bins=num_bins, filter=True, filter_threshold=histogram_density_uncertainty_filter)

        # Update the histogram data in the transfer function
        histogram_dict['histogram_scene'].SetInputData(helpers.vtk_create_table(hist_norm_opacity), 0, 1)
        histogram_dict['histogram_color'].SetInputData(helpers.vtk_create_table(hist_norm_color), 0, 1)
        histogram_dict['histogram_density'].SetInputData(helpers.vtk_create_table(hist_norm_density), 0, 1)
        
        # Redraw the histogram
        # interactor_dict['interactor_tf_opacity'].GetRenderWindow().Render()
        interactor_dict['interactor_opacity'].GetRenderWindow().Render()
        interactor_dict['interactor_tf_color_uncertainty'].GetRenderWindow().Render()
        interactor_dict['interactor_tf_density_uncertainty'].GetRenderWindow().Render()

    spin_box.valueChanged.connect(update_histogram)

    
    """
    Selected points from the scatter plot (uncertainties color and density)
    """
    data_colorUncertainty = color_uncertainty_volume.GetPointData().GetScalars()
    npoints_colorUncertainty = color_uncertainty_volume.GetNumberOfPoints()
    npoints_colorUncertainty_ind = np.arange(npoints_colorUncertainty)
    # data_colorUncertainty_numpy = numpy_support.vtk_to_numpy(data_colorUncertainty)

    data_densityUncertainty = density_uncertainty_volume.GetPointData().GetScalars()
    # data_densityUncertainty_numpy = numpy_support.vtk_to_numpy(data_densityUncertainty)

    def selectedInd(ind):
        """
        Update the selected points in the color and density uncertainty volumes.

        Parameters:
        -----------
        ind : array_like
            Indices of the selected points.

        Returns:
        --------
        None

        Notes:
        ------
        This function adjusts the alpha values of the color and density uncertainty volumes 
        based on the selected indices. It sets the alpha values to 0 for unselected points.
        Additionally, it updates the color and density ranges for the histograms, creates 
        rectangles representing the selected range on the histograms, and updates the scene 
        visualization accordingly. If no points are selected, it resets the visualization and 
        removes the rectangles from the histograms.
        """
        if len(ind) > 0:
            # color_uncertainty_volume.GetPointData().SetScalars(alpha)
            # density_uncertainty_volume.GetPointData().SetScalars(alpha)

            alpha_data_colorUncertainty = numpy_support.vtk_to_numpy(data_colorUncertainty)
            copy_alpha_data_colorUncertainty = np.copy(alpha_data_colorUncertainty)

            alpha_data_densityUncertainty = numpy_support.vtk_to_numpy(data_densityUncertainty)
            copy_alpha_data_densityUncertainty = np.copy(alpha_data_densityUncertainty)

            mask = ~np.isin(npoints_colorUncertainty_ind, ind)
            ind_inverse = npoints_colorUncertainty_ind[mask]

            copy_alpha_data_colorUncertainty[ind_inverse] = 0.0
            copy_alpha_data_densityUncertainty[ind_inverse] = 0.0

            alpha_colorUnc = numpy_support.numpy_to_vtk(num_array=copy_alpha_data_colorUncertainty, deep=True)
            alpha_denUnc = numpy_support.numpy_to_vtk(num_array=copy_alpha_data_densityUncertainty, deep=True)

            color_uncertainty_volume.GetPointData().SetScalars(alpha_colorUnc)
            density_uncertainty_volume.GetPointData().SetScalars(alpha_denUnc)

            # update_histogram(spin_box.value(), selected_scatter_pts=True)

            data_color_ind = data_color[ind]
            data_density_ind = data_density[ind]
            color_max_min = np.array([np.min(data_color_ind), np.max(data_color_ind)])
            density_max_min = np.array([np.min(data_density_ind), np.max(data_density_ind)])

            rectangle_color_uncertainy = helpers.vtk_create_rectangle(chart_dict['chart_tf_color_uncertainty'], color_max_min)
            rectangle_density_uncertainty = helpers.vtk_create_rectangle(chart_dict['chart_tf_density_uncertainty'], density_max_min)
            rectangle_dict['rectangle_color_uncertainy'] = rectangle_color_uncertainy
            rectangle_dict['rectangle_density_uncertainty'] = rectangle_density_uncertainty

            zBuffer(None, None)
        else:
            color_uncertainty_volume.GetPointData().SetScalars(data_colorUncertainty)
            density_uncertainty_volume.GetPointData().SetScalars(data_densityUncertainty)

            # update_histogram(spin_box.value(), selected_scatter_pts=False)

            rectangle_color_index = chart_dict['chart_tf_color_uncertainty'].GetPlotIndex(rectangle_dict['rectangle_color_uncertainy'])
            if rectangle_color_index >= 0:
                chart_dict['chart_tf_color_uncertainty'].RemovePlot(rectangle_color_index)
            
            rectangle_density_index = chart_dict['chart_tf_density_uncertainty'].GetPlotIndex(rectangle_dict['rectangle_density_uncertainty'])
            if rectangle_density_index >= 0:
                chart_dict['chart_tf_density_uncertainty'].RemovePlot(rectangle_density_index)
            

            zBuffer(None, None)

        interactor_isosurface.GetRenderWindow().Render()
        interactor_opacity_colorUnc.GetRenderWindow().Render()
        interactor_opacity_densityUnc.GetRenderWindow().Render()

        return None
    

    """
    TAB 1 (1-D transfer functions and 2-D scatter plot)
    """
    def tab1_1d_tf_scatter_plot(tab):
        """
        Create the layout for the 1D transfer function scatter plot tab.

        Notes:
        ------
        This function creates a grid layout for the 1D transfer function scatter plot tab.
        It creates frames for different sections of the layout, including the scatter plot area
        and the control area. It then arranges these frames within the grid layout.
        """
        layout_tab1 = QGridLayout(tab)

        UI_helpers.create_frame('frame_tab1_00', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab1_01', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab1_10', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab1_11', frame_style_sheet, width=500, height=500)

        UI_helpers.create_frame('frame_tab1_01_top', frame_style_sheet, width=500, height=460)
        UI_helpers.create_frame('frame_tab1_01_bottom', frame_style_sheet_border, width=500, height=40)
        UI_helpers.create_layout('layout_frame_tab1_01_bottom', frame=UI_helpers.frame_dict['frame_tab1_01_bottom'], layout_type='QV')

        frame_tab1_01_layout = QVBoxLayout()
        frame_tab1_01_layout.addWidget(UI_helpers.frame_dict['frame_tab1_01_top'])
        frame_tab1_01_layout.addWidget(UI_helpers.frame_dict['frame_tab1_01_bottom'])

        frame_tab1_01_widget = QWidget()
        frame_tab1_01_widget.setLayout(frame_tab1_01_layout)

        layout_tab1.addWidget(UI_helpers.frame_dict['frame_tab1_00'], 0, 0)
        # layout_tab1.addWidget(UI_helpers.frame_dict['frame_tab1_01'], 0, 1)
        layout_tab1.addWidget(frame_tab1_01_widget, 0, 1)
        layout_tab1.addWidget(UI_helpers.frame_dict['frame_tab1_10'], 1, 0)
        layout_tab1.addWidget(UI_helpers.frame_dict['frame_tab1_11'], 1, 1)


        """
        Transfer function (Opacity)
        Histogram opacity for scene: position on the background of scene opacity transfer function
        """
        frame_tab1_01_top = UI_helpers.frame_dict['frame_tab1_01_top']
        interactor_opacity = QVTKRenderWindowInteractor(frame_tab1_01_top)
        interactor_dict['interactor_opacity'] = interactor_opacity
        frame_tab1_01_top.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_tab1_01_top, interactor_opacity)

        # data_opacity, dims_opacity = helpers.vtk_structured_point_value_array(opacity_reader)
        # data_opacity = np.array(opacity_volume.GetPointData().GetScalars())
        hist_norm_opacity = helpers.create_histogram_array(data_opacity, num_bins=20, filter=False)

        view_scene, chart_scene, histogram_scene, isosurface_value = helpers.vtk_create_histogram("Opacity", "Normalized opacity value", 
                                                                                "Scalar value", data=hist_norm_opacity, isosurface=isosurface_filter_value)
        view_scene.SetRenderWindow(interactor_opacity.GetRenderWindow())
        view_dict['view_histogram_scene'] = view_scene
        chart_dict['historgram_chart'] = chart_scene
        histogram_dict['histogram_scene'] = histogram_scene
        isosurface_dict['isosurface_value'] = isosurface_value


        """
        Radio button
        Show or remove the scene geometry.
        If the scene geometry is visible, Z-buffer from the isosurface is used for volume rendering.
        Otherwise, the Z-buffer is omitted.
        """
        layout_frame_tab1_01_bottom = UI_helpers.layout_dict['layout_frame_tab1_01_bottom']

        radio_button_scene_geometry = QRadioButton("Display scene object (vertical line: isosurface value).")
        radio_button_scene_geometry.setChecked(True)
        # radio_button_scene_geometry.setChecked(False)

        layout_radio_button_scene_geometry = QVBoxLayout()
        layout_radio_button_scene_geometry.addWidget(radio_button_scene_geometry)
        layout_frame_tab1_01_bottom.addLayout(layout_radio_button_scene_geometry)

        global radio_button_opacity
        radio_button_opacity = True

        def radioButton_sceneGeometry_onClickced():
            """
            Function to handle the click event of the radio button for scene geometry.

            This function toggles the visibility of the scene geometry (isosurface) and updates the related components 
            accordingly based on the state of the radio button.

            Returns:
            --------
            None
            """
            global radio_button_opacity
            if radio_button_scene_geometry.isChecked():
                radio_button_opacity = True
                isosurface_dict['isosurface_value'].SetInputData(helpers.vtk_create_vertical_line_table(line=True), 0, 1)

                renderer_opacity_colorUnc.AddVolume(volume_opacity)
                renderer_opacity_densityUnc.AddVolume(volume_opacity)

                zBuffer(None, None)

            else:
                radio_button_opacity = False
                isosurface_dict['isosurface_value'].SetInputData(helpers.vtk_create_vertical_line_table(line=False), 0, 1)

                renderer_opacity_colorUnc.RemoveVolume(volume_opacity)
                renderer_opacity_densityUnc.RemoveVolume(volume_opacity)

                zBuffer(None, None)


            # view_dict['view_histogram_scene'].GetRenderWindow().Render()
            interactor_dict['interactor_opacity'].GetRenderWindow().Render()

            interactor_isosurface.GetRenderWindow().Render()
            interactor_opacity_colorUnc.GetRenderWindow().Render()
            interactor_opacity_densityUnc.GetRenderWindow().Render()

        radio_button_scene_geometry.toggled.connect(radioButton_sceneGeometry_onClickced)


        """
        Transfer function (color uncertainty)
        Histogram color uncertainty: position on the background of color uncertainty transfer function
        """
        frame_tab1_00 = UI_helpers.frame_dict['frame_tab1_00']
        interactor_tf_color_uncertainty = QVTKRenderWindowInteractor(frame_tab1_00)
        interactor_dict['interactor_tf_color_uncertainty'] = interactor_tf_color_uncertainty
        frame_tab1_00.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_tab1_00, interactor_tf_color_uncertainty)

        # data_color, dims_color = helpers.vtk_structured_point_value_array(color_uncertainty_reader)
        # data_color = np.array(color_uncertainty_volume.GetPointData().GetScalars())
        hist_norm_color = helpers.create_histogram_array(data_color, num_bins=20, filter=True, filter_threshold=histogram_color_uncertainty_filter)
        # hist_norm_color = np.zeros_like(hist_norm_color)

        view_tf_color_uncertainty, chart_tf_color_uncertainty, item_tf_color_uncertainty, control_points_tf_color_uncertainty, histogram_color = helpers.vtk_create_transfer_function("Color uncertainty", 
                                                                "Optical property (opacity and color)", "Scalar value",
                                                                alpha_tf_color_uncertainty, opacity_tf_color_uncertainty, data=hist_norm_color)
        view_tf_color_uncertainty.SetRenderWindow(interactor_tf_color_uncertainty.GetRenderWindow())
        chart_dict['chart_tf_color_uncertainty'] = chart_tf_color_uncertainty
        histogram_dict['histogram_color'] = histogram_color

        # control_points_tf_color_uncertainty.AddObserver(vtk.vtkCommand.ModifiedEvent, onTransferFunctionPointModified)
        control_points_tf_color_uncertainty.AddObserver(vtk.vtkCommand.EndEvent, onTransferFunctionPointModified)

        """
        Transfer function (density uncertainty)
        Histogram density uncertainty: position on the background of density uncertainty transfer function
        """
        frame_tab1_10 = UI_helpers.frame_dict['frame_tab1_10']
        interactor_tf_density_uncertainty = QVTKRenderWindowInteractor(frame_tab1_10)
        interactor_dict['interactor_tf_density_uncertainty'] = interactor_tf_density_uncertainty
        frame_tab1_10.resizeEvent = lambda event: helpers.vtk_resize_render_window(frame_tab1_10, interactor_tf_density_uncertainty)

        # data_density, dims_density = helpers.vtk_structured_point_value_array(density_uncertainty_reader)
        # data_density = np.array(density_uncertainty_volume.GetPointData().GetScalars())
        hist_norm_density = helpers.create_histogram_array(data_density, num_bins=20, filter=True, filter_threshold=histogram_density_uncertainty_filter)
        # hist_norm_density = np.zeros_like(hist_norm_density)

        view_tf_density_uncertainty, chart_tf_density_uncertainty, item_tf_density_uncertainty, control_points_tf_density_uncertainty, histogram_density = helpers.vtk_create_transfer_function("Density uncertainty",
                                                                    "Optical property (opacity and color)", "Scalar value",
                                                                    alpha_tf_density_uncertainty, opacity_tf_density_uncertainty, data=hist_norm_density)
        view_tf_density_uncertainty.SetRenderWindow(interactor_tf_density_uncertainty.GetRenderWindow())
        chart_dict['chart_tf_density_uncertainty'] = chart_tf_density_uncertainty
        histogram_dict['histogram_density'] = histogram_density

        # control_points_tf_density_uncertainty.AddObserver(vtk.vtkCommand.ModifiedEvent, onTransferFunctionPointModified)
        control_points_tf_density_uncertainty.AddObserver(vtk.vtkCommand.EndEvent, onTransferFunctionPointModified)

        """
        Scatter plot

        Generates a scatter plot based on the color and density uncertainty data
        """
        data_scatter_plot = np.column_stack((data_color, data_density))

        n_data_scatter_plot = data_scatter_plot.shape[0]
        third_column = np.arange(n_data_scatter_plot).astype(int)
        data_scatter_plot_3_columns = np.column_stack((data_scatter_plot, third_column))
        
        # mask
        mask = (data_scatter_plot_3_columns[:,0] <= color_uncertainty_filter) | (data_scatter_plot_3_columns[:,1] <= density_uncertainty_filter)
        filtered_data_scatter_plot_3_columns = data_scatter_plot_3_columns[~mask]

        scatter_plot = ScatterPlot(UI_helpers.frame_dict['frame_tab1_11'], selectedInd, data=filtered_data_scatter_plot_3_columns)
        scatter_plot_dict["scatter_plot"] = scatter_plot
    

    """
    TAB 2 (mean and standard deviation)
    """
    # Shift angles in the y-axis
    angles_shift_y = []
    for i in range(len(angles)):
        angle_y_shifted = angles[i][1] if angles[i][1]==0 or angles[i][1]==360 else 360.0-angles[i][1]
        angles_shift_y.append([angles[i][0], angle_y_shifted])

    # Shift angles in both x and y axes
    angles_list = []
    for i in range(len(angles_shift_y)):
        angle_x_shifted = angles_shift_y[i][0]+180.0 if angles_shift_y[i][0]<180 else angles_shift_y[i][0]-180.0
        angle_y_shifted = angles_shift_y[i][1]+180.0 if angles_shift_y[i][1]<180 else angles_shift_y[i][1]-180.0
        angles_list.append([angle_x_shifted, angle_y_shifted])
    angles_ = np.array(angles_list)

    # Shift heatmap data
    color_means_ = helpers.shift_heatmap(color_means)
    color_standard_deviation_ = helpers.shift_heatmap(color_standard_deviation)
    density_means_ = helpers.shift_heatmap(density_means)
    density_standard_deviation_ = helpers.shift_heatmap(density_standard_deviation)

    # normalize the array for meaningful comparison between heatmaps
    def normalize_array(arr):
        """
        Normalize the array for meaningful comparison between heatmaps.

        Parameters:
        -----------
        arr : numpy.ndarray
            The input array to be normalized.

        Returns:
        --------
        normalized_arr : numpy.ndarray
            The normalized array.

        Notes:
        ------
        This function computes the normalized version of the input array using
        min-max normalization, which scales the values to range between 0 and 1.
        """
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr
    
    color_means_normalize = normalize_array(color_means_)
    color_standard_deviation_normalize = normalize_array(color_standard_deviation_)
    density_means_normalize = normalize_array(density_means_)
    density_standard_deviation_normalize = normalize_array(density_standard_deviation_)

    def tab2_mean_sd(tab):
        """
        Populate tab 2 with frames for displaying mean and standard deviation.

        Parameters:
        -----------
        tab : QWidget
            The tab widget to be populated.

        Returns:
        --------
        None

        Notes:
        ------
        This function creates four frames within the specified tab layout to display
        mean and standard deviation information. It also defines a helper function
        to calculate the similarity between two vectors in terms of angle and cosine similarity.
        """
        layout_tab2 = QGridLayout(tab)

        UI_helpers.create_frame('frame_tab2_00', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab2_01', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab2_10', frame_style_sheet, width=500, height=500)
        UI_helpers.create_frame('frame_tab2_11', frame_style_sheet, width=500, height=500)

        layout_tab2.addWidget(UI_helpers.frame_dict['frame_tab2_00'], 0, 0)
        layout_tab2.addWidget(UI_helpers.frame_dict['frame_tab2_01'], 0, 1)
        layout_tab2.addWidget(UI_helpers.frame_dict['frame_tab2_10'], 1, 0)
        layout_tab2.addWidget(UI_helpers.frame_dict['frame_tab2_11'], 1, 1)

        def similarity_vectors(vector1, vector2):
            """
            Calculate the angle in degrees between two vectors.

            Parameters:
            -----------
            vector1 : numpy.ndarray
                The first vector.
            vector2 : numpy.ndarray
                The second vector.

            Returns:
            --------
            angle : float
                The angle in degrees between the two vectors.
            cos_similarity : float
                The cosine similarity between the two vectors.
            """
            dot_product = np.dot(vector1, vector2)
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            cos_similarity = dot_product / (norm_vector1 * norm_vector2)

            # Calculate the angle in degrees
            angle = np.arccos(cos_similarity) * 180.0 / np.pi

            return angle, cos_similarity
    
        """
        Heatmaps
        """
        def heatmap_motion(event):
            """
            Update the visualization based on mouse movement over the heatmap.

            Parameters:
            -----------
            event : matplotlib.backend_bases.MouseEvent
                Mouse event containing information about the mouse movement.

            Returns:
            --------
            None

            Notes:
            ------
            This function updates the visualization based on the mouse movement over the heatmap.
            It adjusts the camera azimuth and elevation angles to correspond to the mouse position.
            Additionally, it updates the view orientation, preserves depth buffers, and renders the scene.
            Finally, it displays the selected azimuth, elevation, and values from the heatmap on a QLabel.
            """
            # global variable from the radio button which display or remove the scene geometry
            # the radio button is located beneath the histrogram of the scene
            global radio_button_opacity

            if event.xdata is not None and event.ydata is not None:
                y, x = int(event.ydata + 0.5), int(event.xdata + 0.5)
                azimuth, elevation = int(event.ydata + 0.5)*15-180.0, int(event.xdata + 0.5)*15-180.0
                
                helpers.vtk_set_orientation(renderer_isosurface, original_orient)
                helpers.vtk_set_orientation(renderer_opacity_colorUnc, original_orient)
                helpers.vtk_set_orientation(renderer_opacity_densityUnc, original_orient)
                
                camera.Azimuth(azimuth) # east-west
                camera.Elevation(elevation) # north-south

                view_up_vector = camera.GetViewUp()
                view_plane_normal = camera.GetViewPlaneNormal()

                angle, cos_similarity = similarity_vectors(view_up_vector, view_plane_normal)
                if abs(cos_similarity) > 0.95:
                    camera.SetViewUp(0.0, 0.0, 1.0)
                else:
                    camera.SetViewUp(0.0, 1.0, 0.0)
                    
                renderer_isosurface.ResetCamera()
                renderer_opacity_colorUnc.SetActiveCamera(camera)
                renderer_opacity_colorUnc.ResetCamera()
                renderer_opacity_densityUnc.SetActiveCamera(camera)
                renderer_opacity_densityUnc.ResetCamera()

                if radio_button_opacity:
                    renderer_isosurface.PreserveDepthBufferOff()
                    renderer_isosurface.GetRenderWindow().Render()

                    renderer_opacity_colorUnc.PreserveDepthBufferOff()
                    renderer_opacity_colorUnc.GetRenderWindow().Render()

                    renderer_opacity_densityUnc.PreserveDepthBufferOff()
                    renderer_opacity_densityUnc.GetRenderWindow().Render()

                    xmax_isosurface, ymax_isosurface = renderer_isosurface.GetRenderWindow().GetActualSize()
                    renderer_isosurface.GetRenderWindow().GetZbufferData(0, 0, ymax_isosurface-1, xmax_isosurface-1, z_buffer_data_isosurface)

                    xmax_colorUnc, ymax_colorUnc = renderer_opacity_colorUnc.GetRenderWindow().GetActualSize()
                    renderer_opacity_colorUnc.PreserveDepthBufferOn()
                    renderer_opacity_colorUnc.GetRenderWindow().GetZbufferData(0, 0, ymax_colorUnc-1, xmax_colorUnc-1, z_buffer_data_color_uncertainty)
                    renderer_opacity_colorUnc.GetRenderWindow().SetZbufferData(0, 0, ymax_colorUnc-1, xmax_colorUnc-1, z_buffer_data_isosurface)

                    xmax_densityUnc, ymax_densityUnc = renderer_opacity_densityUnc.GetRenderWindow().GetActualSize()
                    renderer_opacity_densityUnc.PreserveDepthBufferOn()
                    renderer_opacity_densityUnc.GetRenderWindow().GetZbufferData(0, 0, ymax_densityUnc-1, xmax_densityUnc-1, z_buffer_data_density_uncertainty)
                    renderer_opacity_densityUnc.GetRenderWindow().SetZbufferData(0, 0, ymax_densityUnc-1, xmax_densityUnc-1, z_buffer_data_isosurface)

                else:
                    renderer_isosurface.PreserveDepthBufferOff()
                    renderer_isosurface.GetRenderWindow().Render()

                    renderer_opacity_colorUnc.PreserveDepthBufferOff()
                    renderer_opacity_colorUnc.GetRenderWindow().Render()

                    renderer_opacity_densityUnc.PreserveDepthBufferOff()
                    renderer_opacity_densityUnc.GetRenderWindow().Render()

                interactor_isosurface.GetRenderWindow().Render()
                interactor_opacity_colorUnc.GetRenderWindow().Render()
                interactor_opacity_densityUnc.GetRenderWindow().Render()

                # Display selected azimuth, elevation and values from the heatmap on the QLabel
                label_heatmap_text = "<h4 style='color:Green;'> Mean and standard deviation (SD) <br> from heatmap </h4> <p> Elevation (Φ): {} <br> Azimuth (θ): {} <br> <br> Color mean: {:.6f} <br> Color SD: {:.6f} <br> <br> Density mean: {:.6f} <br> Density SD: {:.6f} </p>" \
                                    .format(elevation, azimuth, color_means_[y, x], color_standard_deviation_[y, x], density_means_[y, x], density_standard_deviation_[y, x])
                label_heatmap.setText(label_heatmap_text)
            else:
                # print("Clicked outside the heatmap.")
                pass

        # Masking out NaN values in the upper and lower portions of the arrays
        color_means_[0:6, :] = np.nan
        color_means_[19:, :] = np.nan
        color_standard_deviation_[0:6, :] = np.nan
        color_standard_deviation_[19:, :] = np.nan

        # Creating HeatMap objects for displaying mean color uncertainty and standard deviation color uncertainty
        heat_map_color_means = HeatMap(UI_helpers.frame_dict['frame_tab2_00'], data=color_means_, vmin=color_mean_min, vmax=color_mean_max, data_angles=angles_, title="Mean color uncertainty in each direction", color='Reds', file_name="color_uncertainty_means")
        heat_map_color_standard_deviations = HeatMap(UI_helpers.frame_dict['frame_tab2_01'], data=color_standard_deviation_, vmin=color_stddev_min, vmax=color_stddev_max, data_angles=angles_, title="Standard deviation color uncertainty in each direction", color='Reds', file_name="color_uncertainty_SD")

        # Masking out NaN values in the upper and lower portions of the arrays
        density_means_[0:6, :] = np.nan
        density_means_[19:, :] = np.nan
        density_standard_deviation_[0:6, :] = np.nan
        density_standard_deviation_[19:, :] = np.nan

        # Creating HeatMap objects for displaying mean density uncertainty and standard deviation density uncertainty
        heat_map_density_means = HeatMap(UI_helpers.frame_dict['frame_tab2_10'], data=density_means_, vmin=density_mean_min, vmax=density_mean_max, data_angles=angles_, title="Mean density uncertainty in each direction", color='Oranges', file_name="density_uncertainty_means")
        heat_map_density_standard_deviations = HeatMap(UI_helpers.frame_dict['frame_tab2_11'], data=density_standard_deviation_, vmin=density_stddev_min, vmax=density_stddev_max, data_angles=angles_, title="Standard deviation density uncertainty in each direction", color='Oranges', file_name="density_uncertainty_SD")

        # Storing HeatMap objects in a dictionary for easy access
        heatmap_dict['heat_map_color_means'] = heat_map_color_means
        heatmap_dict['heat_map_color_standard_deviations'] = heat_map_color_standard_deviations
        heatmap_dict['heat_map_density_means'] = heat_map_density_means
        heatmap_dict['heat_map_density_standard_deviations'] = heat_map_density_standard_deviations

        # Global variables to store the connection IDs for the motion_notify_event
        global cid_heat_map_color_means
        global cid_heat_map_color_standard_deviations
        global cid_heat_map_density_means
        global cid_heat_map_density_standard_deviations

        # Connect motion_notify_event to the heatmap_motion function for each heatmap
        cid_heat_map_color_means = heat_map_color_means.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
        cid_heat_map_color_standard_deviations = heat_map_color_standard_deviations.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
        cid_heat_map_density_means = heat_map_density_means.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
        cid_heat_map_density_standard_deviations = heat_map_density_standard_deviations.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)

        # Define global variables to keep track of heatmap clicks and coordinates
        global heatmap_click
        heatmap_click = False
        global y_pick, x_pick, azimuth_pick, elevation_pick
        y_pick, x_pick, azimuth_pick, elevation_pick = None, None, None, None

        def heatmap_onpick(event):
            """
            Function is triggered when a point on the heatmap is clicked.

            Parameters:
            -----------
            event : matplotlib event
                The event object containing information about the click.

            Returns:
            --------
            None

            Notes:
            ------
            This function updates the global variables y_pick, x_pick, azimuth_pick, and elevation_pick 
            with the coordinates of the clicked point on the heatmap. It also toggles the heatmap_click 
            variable to indicate whether a click has occurred. If a click is within the valid range of 
            azimuth and elevation, it disconnects the motion event listeners and triggers the creation 
            or removal of a rectangle on the heatmap.
            """
            global y_pick, x_pick, azimuth_pick, elevation_pick
            y_pick, x_pick = int(event.ydata + 0.5), int(event.xdata + 0.5)
            azimuth_pick = int(event.ydata + 0.5)*15
            elevation_pick = int(event.xdata + 0.5)*15+270.0 if int(event.xdata + 0.5)*15<90.0 else int(event.xdata + 0.5)*15-90.0
            
            global heatmap_click
            if y_pick>=90/15 and y_pick<=270/15:
                heatmap_click = not heatmap_click  # Toggle the click state

            global cid_heat_map_color_means
            global cid_heat_map_color_standard_deviations
            global cid_heat_map_density_means
            global cid_heat_map_density_standard_deviations

            if y_pick>=90/15 and y_pick<=270/15:
                if heatmap_click:
                    heat_map_color_means.fig.canvas.mpl_disconnect(cid_heat_map_color_means)
                    heat_map_color_standard_deviations.fig.canvas.mpl_disconnect(cid_heat_map_color_standard_deviations)
                    heat_map_density_means.fig.canvas.mpl_disconnect(cid_heat_map_density_means)
                    heat_map_density_standard_deviations.fig.canvas.mpl_disconnect(cid_heat_map_density_standard_deviations)

                    # Create a rectangle on the heatmap
                    heatmap_selection_square(x_pick, y_pick)

                else:
                    cid_heat_map_color_means = heat_map_color_means.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
                    cid_heat_map_color_standard_deviations = heat_map_color_standard_deviations.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
                    cid_heat_map_density_means = heat_map_density_means.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)
                    cid_heat_map_density_standard_deviations = heat_map_density_standard_deviations.fig.canvas.mpl_connect('motion_notify_event', heatmap_motion)

                    # Remove the rectangle on the heatmap
                    heatmap_selection_square(x_pick, y_pick)

        # Connect the 'button_press_event' to the heatmap_onpick function for each heatmap figure.
        # When a button is pressed on any of these heatmaps, the heatmap_onpick function will be triggered.
        heat_map_color_means.fig.canvas.mpl_connect('button_press_event', heatmap_onpick)
        heat_map_color_standard_deviations.fig.canvas.mpl_connect('button_press_event', heatmap_onpick)
        heat_map_density_means.fig.canvas.mpl_connect('button_press_event', heatmap_onpick)
        heat_map_density_standard_deviations.fig.canvas.mpl_connect('button_press_event', heatmap_onpick)


    tab1_1d_tf_scatter_plot(tab1_1DTF)
    tab2_mean_sd(tab2_2DTF_mean_sd)


    """
    Modified handler
    """
    def ModifiedHandler(obj, event):
        """
        Modified handler function.
        This function is called when an object is modified, triggering a render update for relevant render windows.

        Parameters:
        -----------
        obj : object
            The object that triggered the modification event.
        event : str
            The type of modification event.

        Returns:
        --------
        None
        """
        interactor_isosurface.GetRenderWindow().Render()
        interactor_opacity_colorUnc.GetRenderWindow().Render()
        interactor_opacity_densityUnc.GetRenderWindow().Render()
        # interactor_dict['interactor_tf_opacity'].GetRenderWindow().Render()
        interactor_dict['interactor_opacity'].GetRenderWindow().Render()
        interactor_dict['interactor_tf_color_uncertainty'].GetRenderWindow().Render()
        interactor_dict['interactor_tf_density_uncertainty'].GetRenderWindow().Render()
    
    # Add observers to render windows for ModifiedEvent, calling ModifiedHandler function for updates
    interactor_isosurface.GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    interactor_opacity_colorUnc.GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    interactor_opacity_densityUnc.GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    # interactor_dict['interactor_tf_opacity'].GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    interactor_dict['interactor_opacity'].GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    interactor_dict['interactor_tf_color_uncertainty'].GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)
    interactor_dict['interactor_tf_density_uncertainty'].GetRenderWindow().AddObserver(vtk.vtkCommand.ModifiedEvent, ModifiedHandler)


    """
    Initiate and start QVTK render window interactor
    """
    window.show()

    # Call the zBuffer function for the first time
    zBuffer(None, None)

    selected_pts, disconnect = scatter_plot_dict["scatter_plot"].select_from_collection(
        scatter_plot_dict["scatter_plot"].ax, scatter_plot_dict["scatter_plot"].scatter_pts)

    def heatmap_selection_square(x, y):
        """
        Function to create a selection square on heatmaps.

        Parameters:
        -----------
        x : int
            X-coordinate of the clicked point on the heatmap.
        y : int
            Y-coordinate of the clicked point on the heatmap.

        Returns:
        --------
        None

        Notes:
        ------
        This function triggers the creation of a selection square on each heatmap by calling their respective
        selection_square methods. The x and y coordinates determine the position of the square on the heatmap.
        """
        heatmap_dict['heat_map_color_means'].selection_square(x, y, heatmap_dict['heat_map_color_means'].ax)
        heatmap_dict['heat_map_color_standard_deviations'].selection_square(x, y, heatmap_dict['heat_map_color_standard_deviations'].ax)
        heatmap_dict['heat_map_density_means'].selection_square(x, y, heatmap_dict['heat_map_density_means'].ax)
        heatmap_dict['heat_map_density_standard_deviations'].selection_square(x, y, heatmap_dict['heat_map_density_standard_deviations'].ax)


    # start event processing
    # Source: https://doc.qt.io/qtforpython/porting_from2.html
    # 'exec_' is deprecated and will be removed in the future.
    # Use 'exec' instead.
    try:
        app.exec()
    except AttributeError:
        app.exec_()


_keysyms_for_ascii = (
    None, None, None, None, None, None, None, None,
    None, "Tab", None, None, None, None, None, None,
    None, None, None, None, None, None, None, None,
    None, None, None, None, None, None, None, None,
    "space", "exclam", "quotedbl", "numbersign",
    "dollar", "percent", "ampersand", "quoteright",
    "parenleft", "parenright", "asterisk", "plus",
    "comma", "minus", "period", "slash",
    "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "colon", "semicolon", "less", "equal", "greater", "question",
    "at", "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W",
    "X", "Y", "Z", "bracketleft",
    "backslash", "bracketright", "asciicircum", "underscore",
    "quoteleft", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "braceleft", "bar", "braceright", "asciitilde", "Delete",
    )

_keysyms = {
    Key.Key_Backspace: 'BackSpace',
    Key.Key_Tab: 'Tab',
    Key.Key_Backtab: 'Tab',
    # Key.Key_Clear : 'Clear',
    Key.Key_Return: 'Return',
    Key.Key_Enter: 'Return',
    Key.Key_Shift: 'Shift_L',
    Key.Key_Control: 'Control_L',
    Key.Key_Alt: 'Alt_L',
    Key.Key_Pause: 'Pause',
    Key.Key_CapsLock: 'Caps_Lock',
    Key.Key_Escape: 'Escape',
    Key.Key_Space: 'space',
    # Key.Key_Prior : 'Prior',
    # Key.Key_Next : 'Next',
    Key.Key_End: 'End',
    Key.Key_Home: 'Home',
    Key.Key_Left: 'Left',
    Key.Key_Up: 'Up',
    Key.Key_Right: 'Right',
    Key.Key_Down: 'Down',
    Key.Key_SysReq: 'Snapshot',
    Key.Key_Insert: 'Insert',
    Key.Key_Delete: 'Delete',
    Key.Key_Help: 'Help',
    Key.Key_0: '0',
    Key.Key_1: '1',
    Key.Key_2: '2',
    Key.Key_3: '3',
    Key.Key_4: '4',
    Key.Key_5: '5',
    Key.Key_6: '6',
    Key.Key_7: '7',
    Key.Key_8: '8',
    Key.Key_9: '9',
    Key.Key_A: 'a',
    Key.Key_B: 'b',
    Key.Key_C: 'c',
    Key.Key_D: 'd',
    Key.Key_E: 'e',
    Key.Key_F: 'f',
    Key.Key_G: 'g',
    Key.Key_H: 'h',
    Key.Key_I: 'i',
    Key.Key_J: 'j',
    Key.Key_K: 'k',
    Key.Key_L: 'l',
    Key.Key_M: 'm',
    Key.Key_N: 'n',
    Key.Key_O: 'o',
    Key.Key_P: 'p',
    Key.Key_Q: 'q',
    Key.Key_R: 'r',
    Key.Key_S: 's',
    Key.Key_T: 't',
    Key.Key_U: 'u',
    Key.Key_V: 'v',
    Key.Key_W: 'w',
    Key.Key_X: 'x',
    Key.Key_Y: 'y',
    Key.Key_Z: 'z',
    Key.Key_Asterisk: 'asterisk',
    Key.Key_Plus: 'plus',
    Key.Key_Minus: 'minus',
    Key.Key_Period: 'period',
    Key.Key_Slash: 'slash',
    Key.Key_F1: 'F1',
    Key.Key_F2: 'F2',
    Key.Key_F3: 'F3',
    Key.Key_F4: 'F4',
    Key.Key_F5: 'F5',
    Key.Key_F6: 'F6',
    Key.Key_F7: 'F7',
    Key.Key_F8: 'F8',
    Key.Key_F9: 'F9',
    Key.Key_F10: 'F10',
    Key.Key_F11: 'F11',
    Key.Key_F12: 'F12',
    Key.Key_F13: 'F13',
    Key.Key_F14: 'F14',
    Key.Key_F15: 'F15',
    Key.Key_F16: 'F16',
    Key.Key_F17: 'F17',
    Key.Key_F18: 'F18',
    Key.Key_F19: 'F19',
    Key.Key_F20: 'F20',
    Key.Key_F21: 'F21',
    Key.Key_F22: 'F22',
    Key.Key_F23: 'F23',
    Key.Key_F24: 'F24',
    Key.Key_NumLock: 'Num_Lock',
    Key.Key_ScrollLock: 'Scroll_Lock',
    }


if __name__ == "__main__":
    print(PyQtImpl)
    NeRFDeltaView()

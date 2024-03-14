from logging.handlers import TimedRotatingFileHandler
import vtk
import numpy as np

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
)

colors = vtk.vtkNamedColors()

def vtk_read_volume_from_file(file_name):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_name)
    reader.Update()
    volume = reader.GetOutput()
    return volume, reader

def vtk_structured_point_value_array(reader):
    reader_output = reader.GetOutput()
    dims = reader_output.GetDimensions()

    # Get the dimensions of the structured points
    # total_points = dims[0] * dims[1] * dims[2]
    # print("Total number of structured points:", total_points)

    array_values = []

    # Print the values for all structured points
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                idx = x + dims[0] * (y + dims[1] * z)
                value = reader_output.GetScalarComponentAsFloat(x, y, z, 0)
                array_values.append(value)
    arr_value = np.array(array_values)
    # print(np.unique(arr_value))

    return arr_value, dims

    """
    In VTK, the data in a vtkStructuredPoints object is stored in a 1D array, 
    but it represents a 3D volume. To access the values of individual structured points in the 3D volume, 
    you need to map their 3D coordinates (x, y, z) to a linear index in the 1D array.

    Here's how it works:

        x, y, and z are the indices of a structured point in the X, Y, and Z dimensions, respectively.
        dims[0], dims[1], and dims[2] are the dimensions of the structured points in the X, Y, and Z directions, respectively.

    The calculation for the linear index is as follows:

        - We start with the x index, which represents the column number in the 2D slice of the volume. 
        We don't need to adjust this index.
        
        - We then move to the y index, which represents the row number in the 2D slice. 
        To account for this, we multiply the y index by the number of columns (dims[0]) in a single slice and add it to the result.
        
        - Finally, we move to the z index, which represents the slice number along the Z direction.
        To account for this, we multiply the z index by the number of elements (dims[0] * dims[1]) in a single slice, 
        and then add it to the result.

    In the end, idx represents the linear index of the structured point in the 1D array that contains the data of the entire 3D volume. 
    By using this index, we can access the value of the structured point in the array using uncertainty.GetScalarComponentAsFloat(x, y, z, 0).
    """

def vtk_create_interactor_style(interactor):
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    return style

def vtk_contour_filter(inputData, filter_value):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(inputData)
    contour.SetValue(0, filter_value)
    return contour

def vtk_poly_data_mapper(inputConnection):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(inputConnection.GetOutputPort())
    mapper.ScalarVisibilityOff()
    return mapper

def vtk_create_actor(mapper, color):
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.SetVisibility(True)
    return actor

def vtk_create_context_view(color):
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(colors.GetColor3d(color))
    view.GetRenderWindow().SetSize(400,400)
    return view

def vtk_create_chart_xy(view):
    chart = vtk.vtkChartXY()
    view.GetScene().AddItem(chart)
    chart.SetShowLegend(False)
    return chart

# def vtk_create_table(num, arrX, arrY):
#     table = vtk.vtkTable()
#     table.AddColumn(arrX)
#     table.AddColumn(arrY)
#     table.SetNumberOfRows(num)
#     # Only for testing purposes
#     inc = 7.5 / (num - 1)
#     for i in range(num):
#         table.SetValue(i, 0, i * inc)
#         table.SetValue(i, 1, i * inc)
#     return table

def vtk_create_float_array(column_name):
    arr = vtk.vtkFloatArray()
    arr.SetName(column_name)
    return arr

def vtk_create_points(chart, table):
    points = chart.AddPlot(vtk.vtkChart.POINTS)
    points.SetInputData(table, 0, 1)
    points.SetColor(0, 0, 0, 255)
    points.SetWidth(1.0)
    points.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
    return points

def vtk_create_piecewise_function(*args):
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    for i in range(len(args[0])):
        opacityTransferFunction.AddPoint(args[0][i][0], args[0][i][1])
    return opacityTransferFunction

def vtk_create_color_transfer_function(*args):
    alphaTransferFunction = vtk.vtkColorTransferFunction()
    if args[0] == "RGB":
        for i in range(len(args[1])):
            alphaTransferFunction.AddRGBPoint(args[1][i][0], args[1][i][1], args[1][i][2], args[1][i][3])
    elif args[0] == "HSV":
            alphaTransferFunction.AddHSVSegment(args[1][0], args[1][1], args[1][2], args[1][3],
                                                args[1][4], args[1][5], args[1][6], args[1][7])
    return alphaTransferFunction

def vtk_volume_property(color_tf, opacity_tf):
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(color_tf)
    volumeProperty.SetScalarOpacity(opacity_tf)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    return volumeProperty

def vtk_volume_ray_cast_mapper(reader):
    volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())
    return volumeMapper

def vtk_create_volume(volume_mapper, volume_property):
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume

def vtk_create_renderer(*args):
    renderer = vtk.vtkRenderer()
    for i in range(len(args)):
        # renderer.AddVolume(args[i])
        try:
            renderer.AddVolume(args[i])
        except TypeError:
            renderer.AddActor(args[i])
    renderer.SetBackground(colors.GetColor3d('White'))
    return renderer

def vtk_create_outline(reader):
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(reader.GetOutputPort())
    mapperOutline = vtk.vtkPolyDataMapper()
    mapperOutline.SetInputConnection(outline.GetOutputPort())

    actorOutline = vtk.vtkActor()
    actorOutline.SetMapper(mapperOutline)
    actorOutline.GetProperty().SetColor(colors.GetColor3d('Black'))
    return actorOutline

def vtk_create_scalar_bar(tf_uncertainty, interactor, title):
    labelFormat = vtk.vtkTextProperty()
    labelFormat.SetColor(0.0, 0.0, 0.0)

    titleFormat = vtk.vtkTextProperty()
    titleFormat.SetColor(0.0, 0.0, 0.0)

    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetOrientationToHorizontal()
    scalar_bar.SetLookupTable(tf_uncertainty)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    scalar_bar.SetUnconstrainedFontSize(True)
    scalar_bar.SetTitle(title)
    scalar_bar.SetLabelTextProperty(labelFormat)
    scalar_bar.SetTitleTextProperty(titleFormat)
    # scalar_bar.SetPosition(0.50, 0.50)

    scalar_bar_widget = vtk.vtkScalarBarWidget()
    scalar_bar_widget.SetInteractor(interactor)
    scalar_bar_widget.SetScalarBarActor(scalar_bar)
    scalar_bar_widget.On()
    return scalar_bar, scalar_bar_widget

def vtk_create_transfer_function(title, x_axis_title, y_axis_title, color_tf, opacity_tf, data):
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
    view.GetRenderWindow().SetMultiSamples(0)

    chart = vtk.vtkChartXY()
    chart.SetTitle(title)
    chart.GetAxis(0).SetTitle(x_axis_title)
    chart.GetAxis(1).SetTitle(y_axis_title)
    chart.ForceAxesToBoundsOn()
    view.GetScene().AddItem(chart)

    item = vtk.vtkCompositeTransferFunctionItem()
    item.SetColorTransferFunction(color_tf)
    item.SetOpacityFunction(opacity_tf)
    item.SetMaskAboveCurve(True)
    chart.AddPlot(item)

    control_points = vtk.vtkCompositeControlPointsItem()
    control_points.SetColorTransferFunction(color_tf)
    control_points.SetOpacityFunction(opacity_tf)
    chart.AddPlot(control_points)


    color_series = vtk.vtkColorSeries()
    color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3)

    color = color_series.GetColor(0)
    alpha = 10   # opacity
    semi_transparent_color = vtk.vtkColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), int(alpha))

    # Create a table with some points in it...
    table = vtk.vtkTable()

    arr_index = vtk.vtkFloatArray()
    arr_index.SetName("Index")
    table.AddColumn(arr_index)
    
    arr_value = vtk.vtkFloatArray()
    arr_value.SetName("Value")
    table.AddColumn(arr_value)

    table.SetNumberOfRows(len(data))
    num_row = len(data)
    for i in range(num_row):
        table.SetValue(i, 0, (i + 1)/num_row)
        table.SetValue(i, 1, data[i])
    
    line = chart.AddPlot(vtk.vtkChart.BAR)
    line.SetColor(semi_transparent_color.GetRed(),
            semi_transparent_color.GetGreen(),
            semi_transparent_color.GetBlue(),
            semi_transparent_color.GetAlpha())
    line.SetInputData(table, 0, 1)

    yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
    yAxis.SetBehavior(vtk.vtkAxis.FIXED)

    return view, chart, item, control_points, line


def vtk_create_transfer_function_histogram(title, x_axis_title, y_axis_title, color_tf, opacity_tf, data, isosurface):
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
    view.GetRenderWindow().SetMultiSamples(0)

    chart = vtk.vtkChartXY()
    chart.SetTitle(title)
    chart.GetAxis(0).SetTitle(x_axis_title)
    chart.GetAxis(1).SetTitle(y_axis_title)
    chart.ForceAxesToBoundsOn()
    view.GetScene().AddItem(chart)

    item = vtk.vtkCompositeTransferFunctionItem()
    item.SetColorTransferFunction(color_tf)
    item.SetOpacityFunction(opacity_tf)
    item.SetMaskAboveCurve(True)
    chart.AddPlot(item)

    control_points = vtk.vtkCompositeControlPointsItem()
    control_points.SetColorTransferFunction(color_tf)
    control_points.SetOpacityFunction(opacity_tf)
    chart.AddPlot(control_points)


    color_series = vtk.vtkColorSeries()
    color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3)

    color = color_series.GetColor(0)
    alpha = 10   # opacity
    semi_transparent_color = vtk.vtkColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), int(alpha))

    # Create a table with some points in it...
    table = vtk.vtkTable()

    arr_index = vtk.vtkFloatArray()
    arr_index.SetName("Index")
    table.AddColumn(arr_index)
    
    arr_value = vtk.vtkFloatArray()
    arr_value.SetName("Value")
    table.AddColumn(arr_value)

    table.SetNumberOfRows(len(data))
    num_row = len(data)
    for i in range(num_row):
        table.SetValue(i, 0, (i + 1)/num_row)
        table.SetValue(i, 1, data[i])
    
    line = chart.AddPlot(vtk.vtkChart.BAR)
    # line.SetColor(color_series.GetColor(0).GetRed() / 255.0,
    #           color_series.GetColor(0).GetGreen() / 255.0,
    #           color_series.GetColor(0).GetBlue() / 255.0)
    line.SetColor(semi_transparent_color.GetRed(),
            semi_transparent_color.GetGreen(),
            semi_transparent_color.GetBlue(),
            semi_transparent_color.GetAlpha())
    line.SetInputData(table, 0, 1)

    # xAxis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
    # xAxis.SetTitle("")
    # xAxis.GetLabelProperties().SetOpacity(0)
    # xAxis.GetGridPen().SetOpacity(0)
    # xAxis.SetVisible(False)

    yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
    # yAxis.SetTitle("")
    # yAxis.GetLabelProperties().SetOpacity(0)
    # yAxis.GetGridPen().SetOpacity(0)
    # yAxis.SetVisible(False)
    # yAxis.SetRange(0.0, 1.0)
    yAxis.SetBehavior(vtk.vtkAxis.FIXED)

    # chart.AddPlot(item)
    # chart.AddPlot(control_points)

    # Add a vertical line at position 0.90
    table_vertical_line_points = vtk.vtkTable()

    vertical_line_points_index = vtk.vtkFloatArray()
    vertical_line_points_index.SetName("Index")
    table_vertical_line_points.AddColumn(vertical_line_points_index)

    vertical_line_points_value = vtk.vtkFloatArray()
    vertical_line_points_value.SetName("Value")
    table_vertical_line_points.AddColumn(vertical_line_points_value)

    table_vertical_line_points.SetNumberOfRows(2)
    table_vertical_line_points.SetValue(0, 0, isosurface)
    table_vertical_line_points.SetValue(0, 1, 0.0)
    table_vertical_line_points.SetValue(1, 0, isosurface)
    table_vertical_line_points.SetValue(1, 1, 1.0)

    vertical_line = chart.AddPlot(vtk.vtkChart.LINE)
    vertical_line.SetInputData(table_vertical_line_points, 0, 1)
    # vertical_line.SetColor(color_series.GetColor(0).GetRed() / 255.0,
    #                        color_series.GetColor(0).GetGreen() / 255.0,
    #                        color_series.GetColor(0).GetBlue() / 255.0)
    vertical_line.SetColor(0.0, 0.0, 1.0)
    vertical_line.SetWidth(2)

    return view, chart, item, control_points, line, vertical_line

# def vtk_get_transfer_function_control_points(color_tf, opacity_tf):
#     # Get the number of control points in the color and opacity transfer functions
#     num_color_points = color_tf.GetSize()
#     num_opacity_points = opacity_tf.GetSize()

#     # Create lists to store the control points
#     color_control_points = []
#     opacity_control_points = []

#     # Extract the control points from the color transfer function
#     for i in range(num_color_points):
#         color_point = color_tf.GetNodeValue(i)
#         color_control_points.append(color_point)

#     # Extract the control points from the opacity transfer function
#     for i in range(num_opacity_points):
#         opacity_point = opacity_tf.GetNodeValue(i)
#         opacity_control_points.append(opacity_point)

#     return color_control_points, opacity_control_points

def vtk_create_table(data):
    table = vtk.vtkTable()
        
    arr_index = vtk.vtkFloatArray()
    arr_index.SetName("Index")
    table.AddColumn(arr_index)
    
    arr_value = vtk.vtkFloatArray()
    arr_value.SetName("Value")
    table.AddColumn(arr_value)
    
    table.AddColumn(arr_index)
    table.AddColumn(arr_value)
    
    table.SetNumberOfRows(len(data))
    num_row = len(data)
    for i in range(num_row):
        table.SetValue(i, 0, (i + 1) / num_row)
        table.SetValue(i, 1, data[i])

    return table

def vtk_create_text_actor(interactor, title_text):
    title = vtk.vtkTextActor()
    title.SetTextScaleModeToNone()
    title.GetTextProperty().SetFontSize(24)
    title.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    title.GetTextProperty().SetJustificationToCentered()
    # title.SetPosition(interactor.GetRenderWindow().GetSize()[0] // 2, 0)
    title.SetPosition(200, 0)
    title.SetInput(title_text)
    return title

def vtk_figure_reader(figure_file_name):
    image = vtk.vtkPNGReader()
    image.SetFileName(figure_file_name)
    image.Update()
    texture = vtk.vtkTexture()
    texture.SetInputConnection(image.GetOutputPort())
    return image

def vtk_create_plane(mapper):
    plane = vtk.vtkPlane()
    plane.SetOrigin(mapper.GetCenter())
    plane.SetNormal(1, 0, 0)
    return plane

def vtk_create_plane_widget(interactor):
    plane_widget = vtk.vtkImagePlaneWidget()
    plane_widget.SetInteractor(interactor)
    plane_widget.TextureVisibilityOff()
    plane_widget.UpdatePlacement()
    plane_widget.On()
    return plane_widget

# def vtk_create_histogram(data, view, chart):
#     color_series = vtk.vtkColorSeries()
#     color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3)

#     table = vtk.vtkTable()

#     arr_index = vtk.vtkFloatArray()
#     arr_index.SetName("Index")
#     table.AddColumn(arr_index)
    
#     arr_value = vtk.vtkFloatArray()
#     arr_value.SetName("Value")
#     table.AddColumn(arr_value)

#     table.SetNumberOfRows(len(data))
#     num_row = len(data)
#     for i in range(num_row):
#         table.SetValue(i, 0, (i + 1)/num_row)
#         table.SetValue(i, 1, data[i])
    
#     line = chart.AddPlot(vtk.vtkChart.BAR)
#     line.SetColor(color_series.GetColor(0).GetRed() / 255.0,
#             color_series.GetColor(0).GetGreen() / 255.0,
#             color_series.GetColor(0).GetBlue() / 255.0)
#     line.SetInputData(table, 0, 1)

#     xAxis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
#     xAxis.SetTitle("")
#     xAxis.GetLabelProperties().SetOpacity(0)
#     xAxis.GetGridPen().SetOpacity(0)
#     xAxis.SetVisible(False)

#     yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
#     yAxis.SetTitle("")
#     yAxis.GetLabelProperties().SetOpacity(0)
#     yAxis.GetGridPen().SetOpacity(0)
#     yAxis.SetVisible(False)
#     # yAxis.SetRange(0.0, 1.0)
#     yAxis.SetBehavior(vtk.vtkAxis.FIXED)

#     return view

def vtk_create_histogram(title, x_axis_title, y_axis_title, data, isosurface):
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
    view.GetRenderWindow().SetMultiSamples(0)

    chart = vtk.vtkChartXY()
    chart.SetTitle(title)
    chart.GetAxis(0).SetTitle(x_axis_title)
    chart.GetAxis(1).SetTitle(y_axis_title)
    chart.ForceAxesToBoundsOn()
    view.GetScene().AddItem(chart)

    color_series = vtk.vtkColorSeries()
    color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3)

    color = color_series.GetColor(0)
    alpha = 10   # opacity
    semi_transparent_color = vtk.vtkColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), int(alpha))

    # Create a table with some points in it...
    table = vtk.vtkTable()

    arr_index = vtk.vtkFloatArray()
    arr_index.SetName("Index")
    table.AddColumn(arr_index)
    
    arr_value = vtk.vtkFloatArray()
    arr_value.SetName("Value")
    table.AddColumn(arr_value)

    table.SetNumberOfRows(len(data))
    num_row = len(data)
    for i in range(num_row):
        table.SetValue(i, 0, (i + 1)/num_row)
        table.SetValue(i, 1, data[i])
    
    line = chart.AddPlot(vtk.vtkChart.BAR)
    # line.SetColor(color_series.GetColor(0).GetRed() / 255.0,
    #           color_series.GetColor(0).GetGreen() / 255.0,
    #           color_series.GetColor(0).GetBlue() / 255.0)
    line.SetColor(semi_transparent_color.GetRed(),
            semi_transparent_color.GetGreen(),
            semi_transparent_color.GetBlue(),
            semi_transparent_color.GetAlpha())
    line.SetInputData(table, 0, 1)

    # Set custom tick positions
    ticks_position = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_tick_position = vtk.vtkDoubleArray()
    x_tick_label = vtk.vtkStringArray()
    for i, value in enumerate(ticks_position):
        x_tick_position.InsertNextValue(value)
        x_tick_label.InsertNextValue(str(value))
    
    # ticks_label = [str(tick) for tick in ticks_position]

    xAxis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
    # xAxis.SetTitle("")
    # xAxis.GetLabelProperties().SetOpacity(0)
    # xAxis.GetGridPen().SetOpacity(0)
    # xAxis.SetVisible(False)
    xAxis.SetRange(0.0, 1.0)
    xAxis.SetCustomTickPositions(x_tick_position, x_tick_label)
    # xAxis.SetNumberOfTicks(2)
    xAxis.SetMinimumLimit(0.0)
    xAxis.SetMaximumLimit(1.0)
    xAxis.RecalculateTickSpacing()

    yAxis = chart.GetAxis(vtk.vtkAxis.LEFT)
    # yAxis.SetTitle("")
    # yAxis.GetLabelProperties().SetOpacity(0)
    # yAxis.GetGridPen().SetOpacity(0)
    # yAxis.SetVisible(False)
    # yAxis.SetRange(0.0, 1.0)
    yAxis.SetBehavior(vtk.vtkAxis.FIXED)

    # Add a vertical line at position 0.90
    table_vertical_line_points = vtk.vtkTable()
    
    vertical_line_points_index = vtk.vtkFloatArray()
    vertical_line_points_index.SetName("Index")
    table_vertical_line_points.AddColumn(vertical_line_points_index)

    vertical_line_points_value = vtk.vtkFloatArray()
    vertical_line_points_value.SetName("Value")
    table_vertical_line_points.AddColumn(vertical_line_points_value)
    
    table_vertical_line_points.SetNumberOfRows(2)
    table_vertical_line_points.SetValue(0, 0, isosurface)
    table_vertical_line_points.SetValue(0, 1, 0.0)
    table_vertical_line_points.SetValue(1, 0, isosurface)
    table_vertical_line_points.SetValue(1, 1, 1.0)

    vertical_line = chart.AddPlot(vtk.vtkChart.LINE)
    vertical_line.SetInputData(table_vertical_line_points, 0, 1)
    # vertical_line.SetColor(color_series.GetColor(0).GetRed() / 255.0,
    #                        color_series.GetColor(0).GetGreen() / 255.0,
    #                        color_series.GetColor(0).GetBlue() / 255.0)
    vertical_line.SetColor(0.0, 1.0, 0.0)
    vertical_line.SetWidth(2)

    return view, chart, line, vertical_line

def vtk_create_vertical_line_table(line=True):
    table_vertical_line_points = vtk.vtkTable()
    
    vertical_line_points_index = vtk.vtkFloatArray()
    vertical_line_points_index.SetName("Index")
    table_vertical_line_points.AddColumn(vertical_line_points_index)

    vertical_line_points_value = vtk.vtkFloatArray()
    vertical_line_points_value.SetName("Value")
    table_vertical_line_points.AddColumn(vertical_line_points_value)

    if line:
        table_vertical_line_points.SetNumberOfRows(2)
        table_vertical_line_points.SetValue(0, 0, 0.90)
        table_vertical_line_points.SetValue(0, 1, 0.0)
        table_vertical_line_points.SetValue(1, 0, 0.90)
        table_vertical_line_points.SetValue(1, 1, 1.0)
    else:
        table_vertical_line_points.SetNumberOfRows(2)
        table_vertical_line_points.SetValue(0, 0, 0.90)
        table_vertical_line_points.SetValue(0, 1, 0.0)
        table_vertical_line_points.SetValue(1, 0, 0.90)
        table_vertical_line_points.SetValue(1, 1, 0.0)
    
    return table_vertical_line_points

def create_histogram_array(data, num_bins, filter=False, filter_threshold=0.01):
    if filter:
        mask = data <= filter_threshold
        filtered_data = data[~mask]
        # print(np.unique(filtered_data))
        hist, bin_edges = np.histogram(filtered_data, bins=num_bins, density=False, range=(0.0, 1.0))
    else:
        hist, bin_edges = np.histogram(data, bins=num_bins, density=False, range=(0.0, 1.0))
    hist_norm = hist / np.max(hist)
    return hist_norm

def vtk_resize_render_window(frame, interactor):
    size = frame.size()
    interactor.resize(size.width(), size.height())
    interactor.GetRenderWindow().SetSize(size.width(), size.height())
    interactor.update()

def heatmap_onpick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        print(f"Clicked on cell: x={x}, y={y}")
    else:
        print("Clicked outside the heatmap.")

def vtk_get_orientation(ren):
    """
    Get the camera orientation.
    :param ren: The renderer.
    :return: The orientation parameters.
    """
    p = dict()
    camera = ren.GetActiveCamera()
    p['position'] = camera.GetPosition()
    p['focal point'] = camera.GetFocalPoint()
    p['view up'] = camera.GetViewUp()
    p['distance'] = camera.GetDistance()
    p['clipping range'] = camera.GetClippingRange()
    p['orientation'] = camera.GetOrientation()
    return p

def vtk_set_orientation(ren, p):
    """
    Set the orientation of the camera.
    :param ren: The renderer.
    :param p: The orientation parameters.
    :return:
    """
    camera = ren.GetActiveCamera()
    camera.SetPosition(p['position'])
    camera.SetFocalPoint(p['focal point'])
    camera.SetViewUp(p['view up'])
    camera.SetDistance(p['distance'])
    camera.SetClippingRange(p['clipping range'])

def mean_standard_deviation(azimuth_len, elevation_len, render_window):
    # Create a window-to-image filter
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    # Get the image data
    image_data = window_to_image_filter.GetOutput()

    width, height, _ = image_data.GetDimensions()

    # Access pixel data
    arr_pixel_value = []
    for y in range(height):
        for x in range(width):
            pixel_value = image_data.GetScalarComponentAsFloat(x, y, 0, 0)
            arr_pixel_value.append(pixel_value)
    np_arr_pixel_value = np.array(arr_pixel_value) / 255.0
    mean = np.mean(np_arr_pixel_value)
    standard_deviation = np.std(np_arr_pixel_value)

    return mean, standard_deviation

def vtk_create_histogram_highlight(chart, data, bar_indices, alpha):
    color_series = vtk.vtkColorSeries()
    color_series.SetColorScheme(vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3)

    color = color_series.GetColor(0)
    # alpha_setting = 10   # opacity
    alpha_setting = alpha   # opacity
    semi_transparent_color = vtk.vtkColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), int(alpha_setting))

    # Create a table with some points in it...
    table = vtk.vtkTable()

    arr_index = vtk.vtkFloatArray()
    arr_index.SetName("Index")
    table.AddColumn(arr_index)
    
    arr_value = vtk.vtkFloatArray()
    arr_value.SetName("Value")
    table.AddColumn(arr_value)

    # if len(bar_indices) > 0:
    #     mask = np.isin(np.arange(len(data)), bar_indices)
    #     data[~mask] = 0.0

    table.SetNumberOfRows(len(data))
    num_row = len(data)
    for i in range(num_row):
        table.SetValue(i, 0, (i + 1)/num_row)
        table.SetValue(i, 1, data[i])
    
    line = chart.AddPlot(vtk.vtkChart.BAR)
    line.SetColor(semi_transparent_color.GetRed() * 0,
            semi_transparent_color.GetGreen() * 0,
            semi_transparent_color.GetBlue(),
            semi_transparent_color.GetAlpha())
    line.SetInputData(table, 0, 1)


    return line

def vtk_create_rectangle(chart, data):
    table = vtk.vtkTable()

    arrX = vtk.vtkFloatArray()
    arrX.SetName("X Axis")
    table.AddColumn(arrX)

    arrTop = vtk.vtkFloatArray()
    arrTop.SetName("TopLine")
    table.AddColumn(arrTop)

    arrBottom = vtk.vtkFloatArray()
    arrBottom.SetName("BottomLine")
    table.AddColumn(arrBottom)

    validMask = vtk.vtkCharArray()
    validMask.SetName("ValidMask")
    table.AddColumn(validMask)

    table.SetNumberOfRows(3)

    table.SetValue(0, 0, data[0])
    table.SetValue(0, 1, 1.0)
    table.SetValue(0, 2, 0.0)
    validMask.SetValue(0, '1')

    table.SetValue(1, 0, np.mean(data))
    table.SetValue(1, 1, 1.0)
    table.SetValue(1, 2, 0.0)
    validMask.SetValue(1, '1')

    table.SetValue(2, 0, data[1])
    table.SetValue(2, 1, 1.0)
    table.SetValue(2, 2, 0.0)
    validMask.SetValue(2, '1')

    # Add multiple line plots, setting the colors etc.
    color3d = vtk.vtkNamedColors().GetColor3d("blue")
    area = chart.AddPlot(vtk.vtkChart.AREA)
    area.SetInputData(table)
    area.SetInputArray(0, "X Axis")
    area.SetInputArray(1, "TopLine")
    area.SetInputArray(2, "BottomLine")
    area.SetValidPointMaskName("ValidMask")
    area.GetBrush().SetColorF(color3d.GetRed(), color3d.GetGreen(), color3d.GetBlue(), 0.2)

    return area

def shift_heatmap(mat):
    x_shift = np.zeros_like(mat)
    x_shift_ = np.hstack((mat[:, int(np.floor(mat.shape[1]/2)):mat.shape[1]], mat[:, 1:int(np.floor(mat.shape[1]/2))]))
    x_shift[:, :-1] = x_shift_
    x_shift[:, -1] = x_shift[:, 0]
    mat_shift = np.zeros_like(mat)
    y_shift_ = np.vstack((x_shift[int(np.floor(mat.shape[1]/2)):mat.shape[0], :], x_shift[1:int(np.floor(mat.shape[1]/2)), :]))
    mat_shift[:-1, :] = y_shift_
    mat_shift[-1, :] = mat_shift[0, :]
    return mat_shift
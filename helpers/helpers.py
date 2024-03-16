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
    """
    Read a volume dataset from a file using VTK.

    Parameters:
    -----------
    file_name : str
        The path to the volume file.

    Returns:
    --------
    volume : vtk.vtkStructuredPoints
        The volume dataset read from the file.
    reader : vtk.vtkStructuredPointsReader
        The VTK reader object used to read the volume dataset.
    """
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_name)
    reader.Update()
    volume = reader.GetOutput()
    return volume, reader

def vtk_structured_point_value_array(reader):
    """
    Extract the scalar values from a VTK structured points dataset.

    Parameters:
    -----------
    reader : vtk.vtkStructuredPointsReader
        The VTK reader object containing the structured points dataset.

    Returns:
    --------
    arr_value : numpy.ndarray
        An array containing the scalar values extracted from the dataset.
    dims : tuple
        A tuple containing the dimensions of the structured points dataset.
    """
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
    """
    Create and set an interactor style for a VTK interactor.

    Parameters:
    -----------
    interactor : vtk.vtkRenderWindowInteractor
        The VTK interactor object to which the style will be applied.

    Returns:
    --------
    style : vtk.vtkInteractorStyleTrackballCamera
        The created interactor style.
    """
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    return style

def vtk_contour_filter(inputData, filter_value):
    """
    Create a contour filter for a VTK dataset.

    Parameters:
    -----------
    inputData : vtk.vtkDataSet
        The VTK dataset to which the contour filter will be applied.
    filter_value : float
        The value at which the contour will be generated.

    Returns:
    --------
    contour : vtk.vtkContourFilter
        The contour filter object.
    """
    contour = vtk.vtkContourFilter()
    contour.SetInputData(inputData)
    contour.SetValue(0, filter_value)
    return contour

def vtk_poly_data_mapper(inputConnection):
    """
    Create a poly data mapper for visualization of VTK dataset.

    Parameters:
    -----------
    inputConnection : vtk.vtkAlgorithmOutput
        The output connection of the VTK algorithm.

    Returns:
    --------
    mapper : vtk.vtkPolyDataMapper
        The poly data mapper object.
    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(inputConnection.GetOutputPort())
    mapper.ScalarVisibilityOff()
    return mapper

def vtk_create_actor(mapper, color):
    """
    Create an actor for visualization of VTK dataset.

    Parameters:
    -----------
    mapper : vtk.vtkMapper
        The mapper object for the actor.
    color : tuple or list
        The RGB color for the actor.

    Returns:
    --------
    actor : vtk.vtkActor
        The actor object.
    """
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.SetVisibility(True)
    return actor

def vtk_create_context_view(color):
    """
    Create a context view for visualization.

    Parameters:
    -----------
    color : tuple or list
        The RGB color for the background of the view.

    Returns:
    --------
    view : vtk.vtkContextView
        The context view object.
    """
    view = vtk.vtkContextView()
    view.GetRenderer().SetBackground(colors.GetColor3d(color))
    view.GetRenderWindow().SetSize(400,400)
    return view

def vtk_create_chart_xy(view):
    """
    Create an XY chart and add it to the given context view.

    Parameters:
    -----------
    view : vtk.vtkContextView
        The context view to which the chart will be added.

    Returns:
    --------
    chart : vtk.vtkChartXY
        The XY chart object.
    """
    chart = vtk.vtkChartXY()
    view.GetScene().AddItem(chart)
    chart.SetShowLegend(False)
    return chart

def vtk_create_float_array(column_name):
    """
    Create a float array with the given column name.

    Parameters:
    -----------
    column_name : str
        The name of the column for the float array.

    Returns:
    --------
    arr : vtk.vtkFloatArray
        The float array object.
    """
    arr = vtk.vtkFloatArray()
    arr.SetName(column_name)
    return arr

def vtk_create_points(chart, table):
    """
    Create points plot and add it to the chart.

    Parameters:
    -----------
    chart : vtk.vtkChartXY
        The chart to which the points plot will be added.
    table : vtk.vtkTable
        The input data table.

    Returns:
    --------
    points : vtk.vtkPlotPoints
        The points plot object.
    """
    points = chart.AddPlot(vtk.vtkChart.POINTS)
    points.SetInputData(table, 0, 1)
    points.SetColor(0, 0, 0, 255)
    points.SetWidth(1.0)
    points.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
    return points

def vtk_create_piecewise_function(*args):
    """
    Create a piecewise function for opacity transfer function.

    Parameters:
    -----------
    *args : list of tuples
        Each tuple contains two elements: (opacity_value, intensity_value).

    Returns:
    --------
    opacityTransferFunction : vtk.vtkPiecewiseFunction
        The created opacity transfer function.
    """
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    for i in range(len(args[0])):
        opacityTransferFunction.AddPoint(args[0][i][0], args[0][i][1])
    return opacityTransferFunction

def vtk_create_color_transfer_function(*args):
    """
    Create a color transfer function based on the provided arguments.

    Parameters:
    -----------
    *args : tuple
        args[0] : str
            Specifies the color space (either "RGB" or "HSV").
        args[1] : list of tuples
            If color space is "RGB", each tuple contains four elements: (scalar_value, red_value, green_value, blue_value).
            If color space is "HSV", each tuple contains eight elements: (hue_x1, saturation_x1, value_x1, alpha_x1, hue_x2, saturation_x2, value_x2, alpha_x2).

    Returns:
    --------
    alphaTransferFunction : vtk.vtkColorTransferFunction
        The created color transfer function.
    """
    alphaTransferFunction = vtk.vtkColorTransferFunction()
    if args[0] == "RGB":
        for i in range(len(args[1])):
            alphaTransferFunction.AddRGBPoint(args[1][i][0], args[1][i][1], args[1][i][2], args[1][i][3])
    elif args[0] == "HSV":
            alphaTransferFunction.AddHSVSegment(args[1][0], args[1][1], args[1][2], args[1][3],
                                                args[1][4], args[1][5], args[1][6], args[1][7])
    return alphaTransferFunction

def vtk_volume_property(color_tf, opacity_tf):
    """
    Create a volume property with specified color and opacity transfer functions.

    Parameters:
    -----------
    color_tf : vtk.vtkColorTransferFunction
        The color transfer function.
    opacity_tf : vtk.vtkPiecewiseFunction
        The opacity transfer function.

    Returns:
    --------
    volumeProperty : vtk.vtkVolumeProperty
        The created volume property with specified color and opacity transfer functions.
    """
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(color_tf)
    volumeProperty.SetScalarOpacity(opacity_tf)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    return volumeProperty

def vtk_volume_ray_cast_mapper(reader):
    """
    Create a volume ray cast mapper for rendering volumes.

    Parameters:
    -----------
    reader : vtk.vtkStructuredPointsReader
        The reader containing the volume data.

    Returns:
    --------
    volumeMapper : vtk.vtkOpenGLGPUVolumeRayCastMapper
        The volume ray cast mapper for rendering volumes.
    """
    volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())
    return volumeMapper

def vtk_create_volume(volume_mapper, volume_property):
    """
    Create a volume object for rendering.

    Parameters:
    -----------
    volume_mapper : vtk.vtkOpenGLGPUVolumeRayCastMapper
        The volume ray cast mapper for the volume.
    volume_property : vtk.vtkVolumeProperty
        The volume property specifying rendering properties.

    Returns:
    --------
    volume : vtk.vtkVolume
        The volume object for rendering.
    """
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    return volume

def vtk_create_renderer(*args):
    """
    Create a renderer for visualization.

    Parameters:
    -----------
    *args : vtk.vtkVolume or vtk.vtkActor
        Volume or actor objects to be added to the renderer.

    Returns:
    --------
    renderer : vtk.vtkRenderer
        The created renderer for visualization.
    """
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
    """
    Create an outline actor for the volume.

    Parameters:
    -----------
    reader : vtk.vtkStructuredPointsReader
        Reader object containing the volume data.

    Returns:
    --------
    actorOutline : vtk.vtkActor
        Actor representing the outline of the volume.
    """
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(reader.GetOutputPort())
    mapperOutline = vtk.vtkPolyDataMapper()
    mapperOutline.SetInputConnection(outline.GetOutputPort())

    actorOutline = vtk.vtkActor()
    actorOutline.SetMapper(mapperOutline)
    actorOutline.GetProperty().SetColor(colors.GetColor3d('Black'))
    return actorOutline

def vtk_create_scalar_bar(tf_uncertainty, interactor, title):
    """
    Create a scalar bar widget for visualization.

    Parameters:
    -----------
    tf_uncertainty : vtk.vtkColorTransferFunction
        Color transfer function for uncertainty visualization.
    interactor : vtk.vtkRenderWindowInteractor
        Render window interactor for the scalar bar widget.
    title : str
        Title of the scalar bar.

    Returns:
    --------
    scalar_bar : vtk.vtkScalarBarActor
        Scalar bar actor.
    scalar_bar_widget : vtk.vtkScalarBarWidget
        Scalar bar widget for interactivity.
    """
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
    """
    Create a transfer function visualization.

    Parameters:
    -----------
    title : str
        Title of the transfer function.
    x_axis_title : str
        Title of the x-axis.
    y_axis_title : str
        Title of the y-axis.
    color_tf : vtk.vtkColorTransferFunction
        Color transfer function.
    opacity_tf : vtk.vtkPiecewiseFunction
        Opacity transfer function.
    data : np.ndarray
        Data used for visualization.

    Returns:
    --------
    view : vtk.vtkContextView
        Context view for the transfer function.
    """
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

def vtk_create_table(data):
    """
    Create a VTK table from input data.

    Parameters:
    -----------
    data : list or np.ndarray
        Input data for creating the table.

    Returns:
    --------
    table : vtk.vtkTable
        VTK table containing the input data.
    """
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
    """
    Create a VTK text actor with the specified title text.

    Parameters:
    -----------
    interactor : vtk.vtkRenderWindowInteractor
        VTK render window interactor.
    title_text : str
        Text to be displayed as the title.

    Returns:
    --------
    title : vtk.vtkTextActor
        VTK text actor with the specified title text.
    """
    title = vtk.vtkTextActor()

    # Set text properties
    title.SetTextScaleModeToNone()
    title.GetTextProperty().SetFontSize(24)
    title.GetTextProperty().SetColor(0.0, 0.0, 0.0)
    title.GetTextProperty().SetJustificationToCentered()
    # title.SetPosition(interactor.GetRenderWindow().GetSize()[0] // 2, 0)

    # Set position and input text
    title.SetPosition(200, 0)
    title.SetInput(title_text)
    return title

def vtk_figure_reader(figure_file_name):
    """
    Read a PNG figure file and create a VTK texture.

    Parameters:
    -----------
    figure_file_name : str
        Path to the PNG figure file.

    Returns:
    --------
    image : vtk.vtkPNGReader
        VTK PNG reader object containing the figure.
    """
    image = vtk.vtkPNGReader()  # Create a VTK PNG reader
    image.SetFileName(figure_file_name) # Set the file name
    image.Update() # Update the reader
    texture = vtk.vtkTexture()
    texture.SetInputConnection(image.GetOutputPort())
    return image

def vtk_create_histogram(title, x_axis_title, y_axis_title, data, isosurface):
    """
    Create a histogram using VTK.

    Parameters:
    -----------
    title : str
        Title of the histogram.
    x_axis_title : str
        Title of the x-axis.
    y_axis_title : str
        Title of the y-axis.
    data : list
        List of data points for the histogram.
    isosurface : float
        Value for the isosurface.

    Returns:
    --------
    view : vtk.vtkContextView
        VTK context view.
    chart : vtk.vtkChartXY
        VTK chart object.
    line : vtk.vtkPlotBar
        VTK bar plot object.
    vertical_line : vtk.vtkPlotLine
        VTK line plot object representing the vertical line at the specified isosurface value.
    """
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
    """
    Create a VTK table containing points for a vertical line.

    Parameters:
    -----------
    line : bool, optional
        If True, creates a line spanning the entire y-axis range. If False, creates a single point at y=0.

    Returns:
    --------
    table_vertical_line_points : vtk.vtkTable
        VTK table containing the points for the vertical line.
    """
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
    """
    Create a normalized histogram array from the input data.

    Parameters:
    -----------
    data : array_like
        Input data for which the histogram is to be computed.
    num_bins : int
        Number of bins in the histogram.
    filter : bool, optional
        If True, applies a filter to exclude data points below a certain threshold.
    filter_threshold : float, optional
        Threshold value for the filter.

    Returns:
    --------
    hist_norm : numpy.ndarray
        Normalized histogram array.
    """
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
    """
    Resize the render window and interactor to match the size of the frame.

    Parameters:
    -----------
    frame : QFrame
        The frame whose size is used for resizing.
    interactor : vtkGenericRenderWindowInteractor
        The interactor associated with the render window.
    """
    size = frame.size()
    interactor.resize(size.width(), size.height())
    interactor.GetRenderWindow().SetSize(size.width(), size.height())
    interactor.update()

def heatmap_onpick(event):
    """
    Function to handle picking events on a heatmap.

    Parameters:
    -----------
    event : MouseEvent
        The mouse event object generated by the pick event.
    """
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        print(f"Clicked on cell: x={x}, y={y}")
    else:
        print("Clicked outside the heatmap.")

def vtk_get_orientation(ren):
    """
    Get the camera orientation.

    Parameters:
    -----------
    ren : vtkRenderer
        The renderer.

    Returns:
    --------
    dict
        A dictionary containing the orientation parameters:
        - 'position': The position of the camera.
        - 'focal point': The focal point of the camera.
        - 'view up': The up direction of the camera.
        - 'distance': The distance of the camera from the focal point.
        - 'clipping range': The clipping range of the camera.
        - 'orientation': The orientation of the camera.
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

    Parameters:
    -----------
    ren : vtkRenderer
        The renderer.
    p : dict
        A dictionary containing the orientation parameters:
        - 'position': The position of the camera.
        - 'focal point': The focal point of the camera.
        - 'view up': The up direction of the camera.
        - 'distance': The distance of the camera from the focal point.
        - 'clipping range': The clipping range of the camera.

    Returns:
    --------
    None
    """
    camera = ren.GetActiveCamera()
    camera.SetPosition(p['position'])
    camera.SetFocalPoint(p['focal point'])
    camera.SetViewUp(p['view up'])
    camera.SetDistance(p['distance'])
    camera.SetClippingRange(p['clipping range'])

def mean_standard_deviation(azimuth_len, elevation_len, render_window):
    """
    Calculate the mean and standard deviation of pixel values in the render window.

    Parameters:
    -----------
    azimuth_len : int
        Length of azimuth.
    elevation_len : int
        Length of elevation.
    render_window : vtkRenderWindow
        The render window to extract pixel values from.

    Returns:
    --------
    mean : float
        The mean of pixel values.
    standard_deviation : float
        The standard deviation of pixel values.
    """
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

def vtk_create_rectangle(chart, data):
    """
    Create a rectangle plot on a VTK chart.

    Parameters:
    -----------
    chart : vtkChartXY
        The VTK chart object.
    data : list or numpy.ndarray
        The data points for the rectangle.

    Returns:
    --------
    area : vtkPlot
        The created rectangle plot.
    """
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
    """
    Shift the heatmap matrix by half of its width and height.

    Parameters:
    -----------
    mat : numpy.ndarray
        The input heatmap matrix.

    Returns:
    --------
    mat_shift : numpy.ndarray
        The shifted heatmap matrix.
    """
    x_shift = np.zeros_like(mat)
    x_shift_ = np.hstack((mat[:, int(np.floor(mat.shape[1]/2)):mat.shape[1]], mat[:, 1:int(np.floor(mat.shape[1]/2))]))
    x_shift[:, :-1] = x_shift_
    x_shift[:, -1] = x_shift[:, 0]
    mat_shift = np.zeros_like(mat)
    y_shift_ = np.vstack((x_shift[int(np.floor(mat.shape[1]/2)):mat.shape[0], :], x_shift[1:int(np.floor(mat.shape[1]/2)), :]))
    mat_shift[:-1, :] = y_shift_
    mat_shift[-1, :] = mat_shift[0, :]
    return mat_shift
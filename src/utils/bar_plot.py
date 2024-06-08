# Interactive Plot for the spiking events
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation, Whisker, ColumnDataSource, Range1d, LabelSet
import numpy as np

"""
create_raster_fig: Create a raster figure with the given parameters

Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): List containing the labels for each Bar
    y (list): List containing the counts of each Bar
    x_range (tuple): Range of the x-axis    (Can be used to sort the bars in a specific order)
    sizing_mode (str): Sizing mode of the plot
    bar_width (float): Width of the bars
    is_vertical (bool): If the bars are vertical or horizontal
    height (int): Height of the plot
Returns:
    bplt.Figure: The plot
"""
def create_bar_fig(title, x_axis_label, y_axis_label, 
               x, y, x_range, sizing_mode=None, tooltips=None, 
               bar_width=0.9, is_vertical=True, height=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_range=x_range,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools="pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @top"
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Add the bars to the plot
    if is_vertical:
        p.vbar(x=x, top=y, width=bar_width, color="blue")
    else:
        p.hbar(y=x, right=y, height=bar_width, color="blue")

    # Set the height of the plot
    if height is not None:
        p.plot_height = height

    # Axis settings
    p.xgrid.grid_line_color = None
    p.y_range.start = 0

    # Return the plot
    return p


def create_histogram(title, x, bins, x_range=None, legend_label="Histogram", is_density=False, 
                     sizing_mode=None, x_axis_label = 'x', y_axis_label = "y", height=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools="pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Histogram
    hist, edges = np.histogram(x, density=is_density, bins=bins)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white",
           fill_color="blue", legend_label=legend_label)

    # Set the height of the plot
    if height is not None:
        p.plot_height = height

    # Axis settings
    p.y_range.start = 0
    if x_range is not None:
        p.x_range = x_range

    # Return the plot
    return p


def create_box_plot(title, box_arrays: np.ndarray, sizing_mode=None, 
                    x_axis_labels = None, y_axis_label = "y", height=None):
    """
    create_box_plot: Create a box plot with the given parameters
    @param title: Title of the plot
    @param box_arrays: List of arrays containing the values for each box
    @param sizing_mode: Sizing mode of the plot
    @param x_axis_label: Label of the x-axis
    @param y_axis_label: Label of the y-axis
    @param height: Height of the plot
    """

    # Create the plot
    p = bplt.figure(title=title, 
            background_fill_color="white", y_axis_label=y_axis_label)

    # Define the number of boxes
    n_boxes = len(box_arrays)

    # Store the max_upper value
    max_upper = 0

    quantiles = [0.25, 0.5, 0.75]
    for (box_idx, box_array) in enumerate(box_arrays):
        # Convert the box array to a numpy array
        box_array = np.array(box_array)

        # Calculate the quantiles of the ripple max amplitudes
        ripple_quantiles = np.quantile(box_array, quantiles)

        print(f"ripple_quantiles:  {ripple_quantiles}")

        # Compute the IQR outlier boundaries
        ripple_iqr = ripple_quantiles[2] - ripple_quantiles[0]
        print("Ripple IQR: ", ripple_iqr)

        # Calculate the upper and lower whisker values
        upper = ripple_quantiles[2] + 1.5 * ripple_iqr
        lower = max(ripple_quantiles[0] - 1.5 * ripple_iqr, 0)
        # Update the max_upper value
        max_upper = max(max_upper, upper)

        # Calculate the x-offset for the box
        x_offset = box_idx * 1.0
        # Calculate the position of the box
        x_left = 0.4 + x_offset
        x_right = 0.6 + x_offset

        # Add a box annotation for the IQR of the current box
        box = BoxAnnotation(top=ripple_quantiles[2], bottom=ripple_quantiles[0], left=x_left, right=x_right, 
                            fill_color="green", fill_alpha=0.4,
                            line_color="black", line_alpha=1.0, line_width=1.5)
        p.add_layout(box)

        # Add a line for the median
        p.line([x_left, x_right], [ripple_quantiles[1], ripple_quantiles[1]], line_color="black", line_width=2)

        # Add whiskers
        source = ColumnDataSource(data=dict(values=box_array))
        # Calculate the x-position of the whisker
        x_whisker = 0.5 + x_offset
        upper_whisker = Whisker(source=source, base=x_whisker, upper=upper, lower=lower)
        p.add_layout(upper_whisker)

        # Add Outliers
        # Find the outliers
        upper_outliers = box_array[box_array > upper]
        lower_outliers = box_array[box_array < lower]
        p.circle([0.5] * len(upper_outliers), upper_outliers, size=5, color="red", fill_alpha=0.6)
        p.circle([0.5] * len(lower_outliers), lower_outliers, size=5, color="red", fill_alpha=0.6)

    # Change the axis
    p.x_range = Range1d(0, n_boxes)

    # Calculate the y_min value according to the max_upper value
    y_min_range = -max(0.2, max_upper * 0.05)
    p.y_range = Range1d(y_min_range, max_upper * 1.2)

    # Add the x-axis labels
    x_axis_ticks = [0.5 + i for i in range(n_boxes)]
    p.xaxis.ticker = x_axis_ticks
    if x_axis_labels is not None:
        # Override the x-axis labels
        p.xaxis.major_label_overrides = dict(zip(x_axis_ticks, x_axis_labels))

    # Return the plot
    return p
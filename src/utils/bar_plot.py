# Interactive Plot for the spiking events
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation, Whisker, ColumnDataSource, Range1d
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


def create_box_plot(title, values: np.ndarray, sizing_mode=None, 
                    x_axis_label = 'x', y_axis_label = "y", height=None):
    # Calculate the quantiles of the ripple max amplitudes
    quantiles = [0.25, 0.5, 0.75]
    ripple_quantiles = np.quantile(values, quantiles)

    print(f"ripple_quantiles:  {ripple_quantiles}")

    # Compute the IQR outlier boundaries
    ripple_iqr = ripple_quantiles[2] - ripple_quantiles[0]
    print("Ripple IQR: ", ripple_iqr)

    # Calculate the upper and lower whisker values
    upper = ripple_quantiles[2] + 1.5 * ripple_iqr
    lower = max(ripple_quantiles[0] - 1.5 * ripple_iqr, 0)

    p = bplt.figure(title=title, 
            background_fill_color="white", y_axis_label=y_axis_label,)

    # Add a box annotation for the IQR
    box = BoxAnnotation(top=ripple_quantiles[2], bottom=ripple_quantiles[0], left=0.4, right=0.6, 
                        fill_color="green", fill_alpha=0.4,
                        line_color="black", line_alpha=1.0, line_width=1.5)
    p.add_layout(box)

    # Add a line for the median
    p.line([0.4, 0.6], [ripple_quantiles[1], ripple_quantiles[1]], line_color="black", line_width=2)

    # Add whiskers
    source = ColumnDataSource(data=dict(values=values))
    upper_whisker = Whisker(source=source, base=0.5, upper=upper, lower=lower)
    p.add_layout(upper_whisker)

    # Add Outliers
    # Find the outliers
    upper_outliers = values[values > upper]
    lower_outliers = values[values < lower]
    p.circle([0.5] * len(upper_outliers), upper_outliers, size=5, color="red", fill_alpha=0.6)
    p.circle([0.5] * len(lower_outliers), lower_outliers, size=5, color="red", fill_alpha=0.6)

    # Change the axis
    p.x_range = Range1d(0, 1)
    p.y_range = Range1d(-0.1, upper * 1.25)

    # Return the plot
    return p
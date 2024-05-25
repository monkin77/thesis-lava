# Interactive Plot for the spiking events
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BasicTickFormatter

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
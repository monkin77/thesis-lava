# Interactive Plot for the spiking events
# bokeh docs: https://docs.bokeh.org/en/2.4.1/docs/first_steps/first_steps_1.html
import bokeh.plotting as bplt
from bokeh.io import curdoc
from bokeh.models import BoxAnnotation, BasicTickFormatter

# Apply the theme to the plot
curdoc().theme = "light_minimal"  # Can be one of "caliber", "dark_minimal", "light_minimal", "night_sky", "contrast"


"""
create_raster_fig: Create a raster figure with the given parameters

Args:
    title (str): Title of the plot
    x_axis_label (str): Label of the x-axis
    y_axis_label (str): Label of the y-axis
    x (np.ndarray): Array of x values
    y_arrays (list): List of tuples containing the (x, y) pairs for each dot to be plotted
    sizing_mode (str): Sizing mode of the plot
    tools (str): Tools to be added to the plot
    tooltips (str): Tooltips to be added to the plot
    legend_location (str): Location of the legend
    legend_bg_fill_color (str): Background fill color of the legend
    legend_bg_fill_alpha (float): Background fill alpha of the legend
    box_annotation_params (dict): Parameters to create a box annotation
Returns:
    bplt.Figure: The plot
"""
def create_raster_fig(title, x_axis_label, y_axis_label, 
               x, y, dot_size=10, sizing_mode=None, tools=None, 
               tooltips=None, box_annotation_params=None, y_axis_ticker=None):
    # Create the plot
    p = bplt.figure(
        title=title,
        x_axis_label=x_axis_label, 
        y_axis_label=y_axis_label,
        sizing_mode=sizing_mode or "stretch_both",    # Make the plot stretch in both width and height
        tools=tools or "pan, box_zoom, wheel_zoom, hover, undo, redo, zoom_in, zoom_out, reset, save",
        tooltips=tooltips or "Data point @x: @y"
        # y_range=(-0.5, 5)  # Set the range of the y-axis,
    )

    # Add the dots to the plot
    p.dot(x, y, size=dot_size, color="blue", alpha=1)

    # Axis settings
    p.xaxis.formatter = BasicTickFormatter(use_scientific=False)

    # Set y-axis to integers
    if y_axis_ticker is not None:
        p.yaxis.ticker = y_axis_ticker


    # Grid settings
    # p.ygrid.grid_line_color = "red"

    # Add a box annotation
    if box_annotation_params is not None:
        inner_box = BoxAnnotation(
            bottom=box_annotation_params["bottom"], 
            top=box_annotation_params["top"], 
            left=box_annotation_params["left"], 
            right= box_annotation_params["right"], 
            fill_alpha=box_annotation_params["fill_alpha"], 
            fill_color=box_annotation_params["fill_color"]
        )
        p.add_layout(inner_box)

    # Change the number of decimal places on hover
    p.hover.formatters = {'@x': 'numeral', '@y': 'numeral'}
    p.hover.tooltips = [("x", "@x{0.0}"), ("y", "@y{0.0000}")]

    # Return the plot
    return p
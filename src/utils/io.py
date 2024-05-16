from numpy import set_printoptions, uint8, max
from PIL import Image

def preview_np_array(np_array, array_label="Np array", edge_items=5):
    """
    Prints the first and last few elements of a numpy array
    @param np_array: The numpy array to preview
    @param array_label: The label to print before the array
    @param edge_items: The number of items to print at the beginning and end of the array (default 5)
    """
    set_printoptions(threshold=edge_items*2, edgeitems=edge_items)
    print(f"{array_label} Shape: {np_array.shape}.\nPreview: {np_array}")

def make_gif(img_arr, filepath, duration=100, loop=0):
    """
    Makes a gif from an array of images
    @img_arr: The array of images to make the gif from
    @filepath: The path to save the gif
    @duration: The duration of each frame in milliseconds (default 100)
    @loop: The number of times to loop the gif (default 1)
    """
    if max(img_arr) <= 1:
        # Convert to 0-255 scale
        img_arr = img_arr * 255

    frames = [Image.fromarray(uint8(img)) for img in img_arr]
    frame_one = frames[0]

    frame_one.save(filepath, format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=loop, )
    
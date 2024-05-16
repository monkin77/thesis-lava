from numpy import set_printoptions

def preview_np_array(np_array, array_label="Np array", edge_items=5):
    """
    Prints the first and last few elements of a numpy array
    @param np_array: The numpy array to preview
    @param array_label: The label to print before the array
    @param edge_items: The number of items to print at the beginning and end of the array (default 5)
    """
    set_printoptions(threshold=edge_items*2, edgeitems=edge_items)
    print(f"{array_label} Shape: {np_array.shape}.\nPreview: {np_array}")
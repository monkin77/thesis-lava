import numpy as np

def convert_to_numpy_array(val, shape, name = "value", verbose=False):
    """
    Converts a given value to a numpy array if it is not already
    @param val: The value to convert. Can be a scalar, list, or numpy array
    @param shape: The shape of the array to convert to
    @param verbose: Whether to print debug messages

    @return: The value as a numpy array
    @raises ValueError: If the value cannot be converted to a numpy array
    """
    if np.isscalar(val):
        if verbose:
            print(f"{name} is scalar, converting to numpy array")
        # If val is a scalar, create an array filled with that value with shape (n_neurons)
        val = np.full(shape, val)
    elif not isinstance(val, np.ndarray):
        # If val is not a scalar and not a numpy array, try to convert it to a numpy array
        try:
            val = np.array(val)
        except Exception as e:
            raise ValueError(f"Failed to convert {name} to a numpy array. Please ensure it is either a scalar, list, or numpy array.") from e
    
    return val
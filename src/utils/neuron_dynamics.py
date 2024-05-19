import numpy as np

def time_constant_to_fraction(tau):
    """Calculate the (du or dv) value for a given time constant tau."""
    fraction_val = 1 - np.exp(-1/tau)
    return fraction_val
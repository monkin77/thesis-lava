from typing import NamedTuple, Optional
import numpy as np

class SignalToSpikeParameters(NamedTuple):
    '''
    Parameters for the signal to spike conversion
    @signal: np.ndarray: Amplitude of the signal at each time point
    @times: np.ndarray: Array containing the time points of the signal
    @threshold_up: float: Threshold crossing in a rising direction
    @threshold_down: float: Threshold crossing in a falling direction
    @refractory_period: float: Period in which no spike will be generated [same units as time vector] (ms)
    @interpolation_factor: Optional[float]: Upsampling factor, new sampling frequency
    '''
    signal: np.ndarray
    times: np.ndarray
    threshold_up: float
    threshold_down: float
    interpolation_factor: float = 0   # Default value for interpolation factor (0) indicating no interpolation
    refractory_period: float = 0    # Default value for refractory period (0)

class SpikeTrains(NamedTuple):
    '''
    Up and down spike trains received by filtering a signal
    '''
    up: np.array
    down: np.array

def skip_refractory_period(times, idx, refractory_period):
    """
    Returns the index of the next time point after the refractory period
    @times (array): Array containing the time points of the signal
    @idx (int): Current index
    @refractory_period (float): time interval in which no spike will be generated [same units as time vector]

    Returns:
    - int: Index of the next time point after the refractory period
    """
    init_time = times[idx]
    for i in range(idx+1, len(times)):
        if times[i] >= init_time + refractory_period:
            return i    # Return the index of the next time point after the refractory period
    
    # If the refractory period exceeds the time points array, return the index after the last time point indicating the end of the signal
    return len(times)    


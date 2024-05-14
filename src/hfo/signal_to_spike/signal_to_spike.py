import numpy as np
from scipy.interpolate import interp1d
from .sts_utils import SignalToSpikeParameters, SpikeTrains, skip_refractory_period

def signal_to_spike(params: SignalToSpikeParameters):
    """
    This functions retuns two spike trains, when the signal crosses the specified threshold in
    a rising direction (UP spikes) and when it crosses the specified threshold in a falling
    direction (DOWN spikes)
    @times (array): Array containing the time points of the signal
    @signal (array): Amplitude of the signal at each time point
    @interpolation_factor (int): upsampling factor, new sampling frequency
    @thr_up (float): threshold crossing in a rising direction
    @thr_dn (float): threshold crossing in a falling direction
    @refractory_period (float): period in which no spike will be generated [same units as time vector]

    Returns:
    - spikes_up (array): spike train for threshold crossing in a rising direction
    - spikes_dn (array): spike train for threshold crossing in a falling direction
    """
    # This variable keeps track of the last recorded membrane potential where a significant oscillatory behavior was detected
    instant_dc = 0   # Initialize the instant DC (Direct Current) variable to 0. (Amplitude at time t)

    # Initialize the spike trains for UP and DOWN spikes
    spikes_up = []
    spikes_dn = []

    # Define variables for the signal and time points
    times = params.times
    amplitudes = params.signal

    # --------------------------------------------------------------------------------
    # ------- Signal Interpolation to a higher sampling frequency (OPTIONAL) ---------
    # --------------------------------------------------------------------------------
    if (params.interpolation_factor is not None) and (params.interpolation_factor > 1.0):
        print("Interpolating...")
        # Return a function approximation of the signal, such that amplitude(t) = f(t)
        # This is done to interpolate the signal to a higher sampling frequency if needed
        interpolated_signal_func = interp1d(params.times, params.signal)
        # Calculate the new number of data points according to the interpolation factor
        num_data_points = np.round(
            (np.max(params.times) - np.min(params.times)) * params.interpolation_factor
        )
        xnew = np.linspace(np.min(params.times), np.max(params.times), 
                        num=int(num_data_points), endpoint=True)
        # Apply the interpolation function to the new time points array (updated sampling frequency)
        ynew = interpolated_signal_func(xnew)

        # Update the signal and time points
        times = xnew
        amplitudes = ynew

    idx = 0
    while idx < len(times):
        # If the signal increased in amplitude more than the THRESHOLD_UP value (since the last update to instant_dc), detect an UP spike. 
        if amplitudes[idx] - instant_dc > params.threshold_up:
            spikes_up.append(times[idx])    # Add the spike time to the UP spike train

            instant_dc = amplitudes[idx]     # Update the current amplitude to the new value

            idx = skip_refractory_period(times, idx, params.refractory_period)    # Skip the refractory period
        # If the signal decreased in amplitude more than the THRESHOLD_DOWN value (since the last update to instant_dc), detect a DOWN spike.
        elif amplitudes[idx] - instant_dc < params.threshold_down:
            spikes_dn.append(times[idx])    # Add the spike time to the DOWN spike train

            instant_dc = amplitudes[idx]    # Update the current amplitude to the new value

            idx = skip_refractory_period(times, idx, params.refractory_period)    # Skip the refractory period
        else:
            # Move to the next time point without updating the instant_dc (no significant oscillatory behavior detected)
            idx += 1

    return SpikeTrains(up=np.array(spikes_up), down=np.array(spikes_dn))
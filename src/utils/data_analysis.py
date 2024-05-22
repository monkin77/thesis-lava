import numpy as np

def find_spike_times(voltage_arr, current_arr, v_th, dv = 0.5) -> np.ndarray:
    """
    @param voltage_arr: np.ndarray
        A 2D array where each row represents the voltage of each neuron at a given time step
    @param current_arr: np.ndarray
        A 2D array where each row represents the current of each neuron at a given time step
    @param v_th: float
        The threshold voltage of the neurons
    @param dv: float
        The voltage decay at each time step
    This function is used to find the spiking times of the neurons in the voltage array
    
    Returns:
        - spike_times: np.ndarray
            An array of tuples where each tuple contains the time and the neuron index
            of the neuron that spiked
    """
    # Array to store the spike times in a tuple (time, neuron_idx)
    spike_times = []
    
    # Define the minimum voltage of the previous time step to spike
    # v_th * dv is the minimum voltage to spike
    min_voltage_to_spike = v_th * dv
    
    # Iterate the voltage array to find the spikes
    for i in range(len(voltage_arr)):
        # Iterate each channel and check if the voltage is greater than the threshold
        for j in range(len(voltage_arr[i])):
            if voltage_arr[i][j] == 0.0 and i > 0:     # Spike detected
                # Check if the previous voltage was greater than the threshold TODO: Explain better
                # Check if the current is positive (excitatory) caused a spike
                if voltage_arr[i-1][j] > min_voltage_to_spike and current_arr[i][j] > 0:
                    spike_times.append((i, j))
                    break

    return np.array(spike_times)
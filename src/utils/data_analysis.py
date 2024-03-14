

"""
This function is used to find the spiking times of the neurons in the voltage array
"""
def find_spike_times(voltage_arr, v_th):
    # Iterate the voltage array to find the spikes
    spike_times = []
    min_voltage_to_spike = v_th / 2
    for i in range(len(voltage_arr)):
        # Iterate each channel and check if the voltage is greater than the threshold
        for j in range(len(voltage_arr[i])):
            if voltage_arr[i][j] == 0 and i > 0 and voltage_arr[i-1][j] > min_voltage_to_spike:     # Spike detected
                spike_times.append(i)
                break

    return spike_times
# Utility Functions for generating ground truth data for Channel and Network Bursts
import numpy as np

def find_channel_bursts(spike_events, num_spikes_to_burst, max_burst_duration, 
                        min_inter_burst_interval, verbose=False):
    '''
    Find Channel Bursts in the spike events data
    
    Parameters
    ----------
    spike_events : np.ndarray
        List of spike events (time_step, channel_id)
    num_spikes_to_burst : int
        Number of spikes to consider as a burst
    max_burst_duration : int
        Max Time from the first spike in the burst to the last spike in the burst to consider as a burst
    min_inter_burst_interval : int
        Minimum Time between two bursts to consider as different bursts
    verbose : bool


    Returns
    -------
    channel_bursts : dict {channel_id: [list of time steps where the bursts occured]}
    channel_bursts_detailed : dict {channel_id: [ list of lists containing the spikes inside each channel burst ]}
    '''
    channel_bursts = {}
    channel_bursts_detailed = {}

    # Dictionary to store the spikes in [time_step - max_burst_duration, time_step] for each channel
    curr_spikes = {}
    for idx, (spike_time, ch_id) in enumerate(spike_events):
        if verbose:
            print(f"Processing spike {spike_time} from channel {ch_id}")
        # Add the current spike to the dictionary of spikes that are within the max_burst_duration
        curr_ch_spikes = curr_spikes.get(ch_id, [])
        curr_ch_spikes.append(spike_time)
        curr_spikes[ch_id] = curr_ch_spikes


        # Update the current spikes map for each channel 
        for key in list(curr_spikes.keys()):
            # Remove the spikes that are outside the max_burst_duration
            curr_spikes[key] = [ spike_t for spike_t in curr_spikes[key] if (spike_time - spike_t <= max_burst_duration) ]

        # Check if the current channel has enough spikes to consider as a burst
        if len(curr_spikes[ch_id]) >= num_spikes_to_burst:
            # Check if the first spike in the burst is distanced enough from the last burst (min_inter_burst_interval)
            curr_ch_bursts = channel_bursts_detailed.get(ch_id, [])
            if len(curr_ch_bursts) > 0:
                last_ch_burst = curr_ch_bursts[-1][0]
                # print("last ch burst:", last_ch_burst)
                if (curr_spikes[ch_id][0] - last_ch_burst) <= min_inter_burst_interval:
                    # print("Skipping burst as distance < min_inter_burst_interval")
                    continue

            # Add the burst to the list of channel bursts
            curr_ch_bursts.append(curr_spikes[ch_id])
            channel_bursts_detailed[ch_id] = curr_ch_bursts

            # Clear the current spikes for the channel
            curr_spikes[ch_id] = []

        # print(f"curr_spikes: {curr_spikes}")

    # Iterate the dictionary of channel bursts and convert the timestep arrays into a single timestep (final spike of each burst)
    for key in list(channel_bursts_detailed.keys()):
        curr_ch_bursts = channel_bursts_detailed[key]
        curr_ch_bursts_timestamps = list(map(lambda ch_burst: ch_burst[-1], curr_ch_bursts))
        channel_bursts[key] = curr_ch_bursts_timestamps

    return channel_bursts, channel_bursts_detailed


def find_net_bursts(spike_events, num_spikes_to_burst, max_burst_duration, 
                        min_inter_burst_interval, verbose=False):
    '''
    Find Network Bursts in the spike events data of Channel Bursts
    
    Parameters
    ----------
    spike_events : np.ndarray
        List of spike events (time_step, channel_id)
    num_spikes_to_burst : int
        Number of spikes to consider as a burst
    max_burst_duration : int
        Max Time from the first spike in the burst to the last spike in the burst to consider as a burst
    min_inter_burst_interval : int
        Minimum Time between two bursts to consider as different bursts
    verbose : bool


    Returns
    -------
    net_bursts : np.ndarray [time_step]
    net_bursts_detailed : np.ndarray [(time_step, channels in the burst)]
    '''
    net_bursts = []
    net_bursts_detailed = []

    # Set to store the channels that spiked in [time_step - max_burst_duration, time_step]
    # Stores a tuple (time_step, channel_id)
    curr_spiking_channels = {}
    for idx, (spike_time, ch_id) in enumerate(spike_events):
        if verbose:
            print(f"Processing spike {spike_time} from channel {ch_id}")

        # Update the current spikes map for each channel
        remove_ch = []
        for ch in curr_spiking_channels.keys():
            curr_ch_spike_time = curr_spiking_channels[ch]
            # Remove the spikes that are outside the max_burst_duration
            if (spike_time - curr_ch_spike_time > max_burst_duration):
                remove_ch.append(ch)

        # Remove the spikes that are outside the max_burst_duration
        for ch in remove_ch:
            curr_spiking_channels.pop(ch)

        # Add the current spike to the set of spikes that are within the max_burst_duration
        curr_spiking_channels[ch_id] = spike_time
        # curr_spiking_channels.add((spike_time, ch_id))

        # Check if the current time step has enough channels spiking to consider as a network burst
        if len(curr_spiking_channels.keys()) >= num_spikes_to_burst:
            # Check if the first spike in the burst is distanced enough from the last burst (min_inter_burst_interval)
            last_net_burst = None
            if len(net_bursts) > 0:
                last_net_burst = net_bursts[-1]     # Get the last network burst

                # Get the first channel spike of the current network burst
                curr_first_ch_spike = min(list(curr_spiking_channels.values()))
                if (curr_first_ch_spike - last_net_burst) <= min_inter_burst_interval:
                    # Skip the current burst as the distance between the NB and th
                    # print("Skipping burst as distance < min_inter_burst_interval")
                    continue
            
            # Add the burst to the list of network bursts
            # Get the last channel spike of the current network burst
            curr_last_ch_spike = max(list(curr_spiking_channels.values()))
            net_bursts.append(curr_last_ch_spike)
            net_bursts_detailed.append( (curr_last_ch_spike, list(curr_spiking_channels.keys()) ) )

            # Clear the current spikes for the channel
            curr_spiking_channels = {}

        # print(f"curr_spikes: {curr_spikes}")

    return net_bursts, net_bursts_detailed
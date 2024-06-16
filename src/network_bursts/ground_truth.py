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

        
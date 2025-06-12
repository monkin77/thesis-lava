from .input import MarkerType

init_offset = 3400  # 33400      #   
virtual_time_step_interval = 1  # TODO: Check if this should be the time-step value. it is not aligned with the sampling rate of the input data

num_steps = 3000    # 200 # Number of steps to run the simulation

class SNNSimConfig:
    """
    This class is used to store the parameters for the simulation of the SNN.
    @param ground_truth: The ground truth data to be used for classifying the feature neurons. It has length=num_steps, and each element is a 0/1
    representing the existence of a relevant event.
    @param init_offset: The initial offset to be used for the simulation. This is the time the simulation starts from.
    @param virtual_time_step_interval: The time interval between two consecutive time steps in the simulation.
    @param num_steps: The number of steps to run the simulation for.
    """

    def __init__(self, ground_truth, init_offset, virtual_time_step_interval, num_steps):
        self.ground_truth = ground_truth
        self.init_offset = init_offset
        self.virtual_time_step_interval = virtual_time_step_interval
        self.num_steps = num_steps

    def real_time_to_iter(self, real_time):
        """
        This function converts the real time to the iteration in the simulation.
        @param real_time (int): The real time to convert to the iteration.

        @return (int): The iteration in the simulation.
        """
        if real_time < self.init_offset:
            # If the real time is before the simulation starts, return index 0
            return 0
        
        return (real_time - self.init_offset) // self.virtual_time_step_interval

class NeuronClass:
    """Enum for the different classes of Feature Neurons"""
    SILENT = 0
    NOISY = 1
    RIPPLE_DETECTOR = 2
    FR_DETECTOR = 3
    BOTH = 4

def marker_type_to_neuron_class(marker_type: MarkerType) -> NeuronClass:
    """
    This function returns the neuron class for the marker type.
    If the marker type is unknown, it returns the noisy class.
    """
    if marker_type == MarkerType.RIPPLE:
        return NeuronClass.RIPPLE_DETECTOR
    elif marker_type == MarkerType.FAST_RIPPLE:
        return NeuronClass.FR_DETECTOR
    elif marker_type == MarkerType.BOTH:
        return NeuronClass.BOTH
    else:
        return NeuronClass.NOISY
    
import numpy as np
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
def train_test_split_seeg_data(up_spike_evts, down_spike_evts, gt_times,
                                  single_ch_duration, train_single_ch_duration, test_single_ch_duration) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the UP/DN Spike Arrays as well as the GT data into training and testing sets.
    The first 80% of the data of each Individual channel of length (single_ch_duration) will be used for training,
    and the remaining 20% will be used for testing.

    Parameters:
    - up_spike_evts: UP Spike Events for each channel
        Shape: (num_channels, num_up_spikes)
    - down_spike_evts: Down Spike Events for each channel
        Shape: (num_channels, num_down_spikes)
    - gt_times: Ground Truth Times for each channel
        Shape: (num_channels, num_gt_times)
    - single_ch_duration: Duration of a single channel in ms
    - train_single_ch_duration: Duration of the training set for a single channel in ms
    - test_single_ch_duration: Duration of the testing set for a single channel in ms

    Returns:
    - train_up_spike_evts: UP Spike Events for the training set
        Shape: (num_channels, num_train_up_spikes)
    - test_up_spike_evts: UP Spike Events for the testing set
        Shape: (num_channels, num_test_up_spikes)
    - train_down_spike_evts: Down Spike Events for the training set
        Shape: (num_channels, num_train_down_spikes)
    - test_down_spike_evts: Down Spike Events for the testing set
        Shape: (num_channels, num_test_down_spikes)

    '''
    train_up_spike_evts, test_up_spike_evts = [], []
    train_down_spike_evts, test_down_spike_evts = [], []
    train_gt_times, test_gt_times = [], []
    for src_idx in range(len(up_spike_evts)):
        # Get the up and down spike events for the current source
        up_spike_evt = up_spike_evts[src_idx]
        down_spike_evt = down_spike_evts[src_idx]
        cur_gt_times = gt_times[src_idx]

        # Split UP Spikes into train and test sets
        train_up_spike_evt_mask = np.mod(up_spike_evt, single_ch_duration) <= train_single_ch_duration
        test_up_spike_evt_mask = np.mod(up_spike_evt, single_ch_duration) > train_single_ch_duration
        train_up_spike_evt = up_spike_evt[train_up_spike_evt_mask]
        test_up_spike_evt = up_spike_evt[test_up_spike_evt_mask]
        # print(f"train_up_spike_evt_mask: {train_up_spike_evt_mask.shape}, test_up_spike_evt_mask: {test_up_spike_evt_mask.shape}")
        # print(f"train_up_spike_evt: {train_up_spike_evt.shape}, test_up_spike_evt: {test_up_spike_evt.shape}")
        assert train_up_spike_evt.shape[0] + test_up_spike_evt.shape[0] == up_spike_evt.shape[0], "Train and Test sets combined do not match the original event shape!"
        # Split the Down Spikes into train and test sets
        train_dn_spike_evt_mask = np.mod(down_spike_evt, single_ch_duration) <= train_single_ch_duration
        test_dn_spike_evt_mask = np.mod(down_spike_evt, single_ch_duration) > train_single_ch_duration
        train_down_spike_evt = down_spike_evt[train_dn_spike_evt_mask]
        test_down_spike_evt = down_spike_evt[test_dn_spike_evt_mask]
        # print(f"train_down_spike_evt: {train_down_spike_evt.shape}, test_down_spike_evt: {test_down_spike_evt.shape}")
        assert train_down_spike_evt.shape[0] + test_down_spike_evt.shape[0] == down_spike_evt.shape[0], "Train and Test sets combined do not match the original event shape!"

        # Split the ground truth data into train and test sets
        train_gt_data_mask = np.mod(cur_gt_times, single_ch_duration) <= train_single_ch_duration
        test_gt_data_mask = np.mod(cur_gt_times, single_ch_duration) > train_single_ch_duration
        cur_train_gt_times = cur_gt_times[train_gt_data_mask]
        cur_test_gt_times = cur_gt_times[test_gt_data_mask]
        assert cur_train_gt_times.shape[0] + cur_test_gt_times.shape[0] == cur_gt_times.shape[0], "Train and Test sets combined do not match the original ground truth shape!"

        # Update the Timestamps to not consider the 20% of the signal that is not used for training for each individual signal
        train_up_spike_evt = train_up_spike_evt - (train_up_spike_evt // single_ch_duration) * test_single_ch_duration
        train_down_spike_evt = train_down_spike_evt - (train_down_spike_evt // single_ch_duration) * test_single_ch_duration
        cur_train_gt_times = cur_train_gt_times - (cur_train_gt_times // single_ch_duration) * test_single_ch_duration

        test_up_spike_evt = test_up_spike_evt - (1 + (test_up_spike_evt // single_ch_duration)) * train_single_ch_duration
        test_down_spike_evt = test_down_spike_evt - (1 + (test_down_spike_evt // single_ch_duration)) * train_single_ch_duration
        cur_test_gt_times = cur_test_gt_times - (1 + (cur_test_gt_times // single_ch_duration)) * train_single_ch_duration

        # Append the train and test sets to the respective lists
        train_up_spike_evts.append(train_up_spike_evt)
        test_up_spike_evts.append(test_up_spike_evt)
        train_down_spike_evts.append(train_down_spike_evt)
        test_down_spike_evts.append(test_down_spike_evt)
        train_gt_times.append(cur_train_gt_times)
        test_gt_times.append(cur_test_gt_times)

    # Convert the lists to numpy arrays
    train_up_spike_evts = np.array(train_up_spike_evts, dtype=object)
    test_up_spike_evts = np.array(test_up_spike_evts, dtype=object)
    train_down_spike_evts = np.array(train_down_spike_evts, dtype=object)
    test_down_spike_evts = np.array(test_down_spike_evts, dtype=object)
    train_gt_times = np.array(train_gt_times, dtype=object)
    test_gt_times = np.array(test_gt_times, dtype=object)

    return train_up_spike_evts, test_up_spike_evts, train_down_spike_evts, test_down_spike_evts, train_gt_times, test_gt_times


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

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
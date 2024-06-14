import math
import numpy as np

def voltage_pred(time, du, dv, spike_times, record_times,
                 spike_weight = 1.0, u_init = 0, v_init = 0):
    '''
    Calculates an approximation of the voltage of a CUBA LIF Neuron given the parameters at a given time step
    (Ignoring Spikes), i.e. only subthreshold dynamics are considered.
    @param time: the time step at which the voltage is to be calculated
    @param du: The inverse of the synaptic membrane time constant
    @param dv: The inverse of the membrane potential time constant
    @param record_times: The times at which the voltage is to be recorded
    @param spike_times: The times at which the neuron spikes
    @param spike_weight The weight of the spikes

    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage

    @return: Array of the voltage at the recorded times
    '''
    curr_u = u_init
    curr_v = v_init
    recorded_times = []
    for time_step in range(0, time+1, 1):
        # print("Updating time step", time_step)
        spike_inc = spike_weight if time_step in spike_times else 0

        curr_u = curr_u * (1-du) + spike_inc
        curr_v = curr_v * (1-dv) + curr_u

        if time_step in record_times:
            recorded_times.append(curr_v)
    
    return recorded_times


class CUBADynamicsResult:
    '''
    A class to represent an Result of the CUBA Dynamics Analysis
    '''
    def __init__(self, spike_weight, du, dv, bef_spike_v, spike_v, after_spike_v):
        self.spike_weight = spike_weight
        self.du = du
        self.dv = dv
        self.bef_spike_v = bef_spike_v
        self.spike_v = spike_v
        self.after_spike_v = after_spike_v

    def to_key(self):
        return f"{self.spike_weight}_{self.du}_{self.dv}"

    def __str__(self):
        return f"Spike Weight: {self.spike_weight}, du: {self.du}, dv: {self.dv}"

    
    # Define method to print the object as a dictionary
    def to_dict(self):
        return {
            "spike_weight": self.spike_weight,
            "du": self.du,
            "dv": self.dv,
            "before v": self.bef_spike_v,
            "spike v": self.spike_v,
            "after v": self.after_spike_v
        }
    
    # Define method to print the object when printed
    def __repr__(self):
        return str(self.to_dict())
    
    # Define method to compare two objects
    def __eq__(self, other):
        return self.to_key() == other.to_key()

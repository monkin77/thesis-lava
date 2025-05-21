from lava.magma.core.model.py.model import PyLoihiProcessModel  # Processes running on CPU inherit from this class
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort

import numpy as np

class SpikeEventGen(AbstractProcess):
    """Input Process that generates spike events based on the input file

    Args:
        shape (tuple): Shape of the output port
        name (str): Name of the process
        spike_events (np.ndarray): Array of spike events to send
        channel_map (dict): Mapping of the channel id to the output index
        init_offset (int):  Arbitrary offset to start the simulation (in ms)
        virtual_time_step_interval (int): Arbitrary time between time steps (in ms). This is not a real time interval (1000ms = 1s)
    """
    def __init__(self, shape: tuple, spike_events: np.ndarray, name: str,
                channel_map, init_offset=0, virtual_time_step_interval=1 ) -> None:
        super().__init__(name=name)
        self.s_out = OutPort(shape=shape)
        self.spike_events = Var(shape=spike_events.shape, init=spike_events)
        self.init_offset = Var(shape=(1,), init=init_offset)
        self.virtual_time_step_interval = Var(shape=(1,), init=virtual_time_step_interval)

        # Add the channel map to the process parameters because 
        # the dict type is not supported by the LavaPyType
        self.proc_params["channel_map"] = channel_map

@implements(proc=SpikeEventGen, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeEventGenModel(PyLoihiProcessModel):
    """Spike Event Generator Process implementation running on CPU (Python)
   
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)   # IT IS POSSIBLE TO SEND FLOATS AFTER ALL Args:
    """
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    spike_events: np.ndarray = LavaPyType(np.ndarray, np.ndarray)
    init_offset: int = LavaPyType(int, int)
    virtual_time_step_interval: int = LavaPyType(int, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params=proc_params)
        # print("spike events", self.spike_events.__str__())    # TODO: Check why during initialization the variable prints the class, while during run it prints the value
        self.curr_spike_idx = 0     # Index of the next spiking event to send
        
        self.channel_map = self.proc_params["channel_map"]

        # print("SpikeEventGen Model Parameters: ")
        # print("init_offset:", self.init_offset)
        # print("virtual_time_step_interval:", self.virtual_time_step_interval)
        # print("channel_map:", self.channel_map)

    def run_spk(self) -> None:
        spike_data = np.zeros(self.s_out.shape) # Initialize the spike data to 0
        
        # Use random spikes to test it
        # spike_data[0] = np.random.random()  # Only 1 neuron is sending data (spikes)
        # self.curr_spike_idx += 1

        # Use the spike events from the file
        # print("spike events: ", self.spike_events[:5])
        # TODO: For now we are using the time_step as the time measurement that will simulate real-time activity. It would be better to have access to the running time
        
        #print("time step:", self.time_step)
        # If the current simulation time is greater than the next spike event, send a spike
        currTime = self.init_offset + self.time_step*self.virtual_time_step_interval

        spiking_channels = set()   # List of channels that will spike at the same time
        while (self.curr_spike_idx < len(self.spike_events)) and currTime >= self.spike_events[self.curr_spike_idx][0]:  # Check if there are more than 1 spike events to send
            if self.spike_events[self.curr_spike_idx][0] < self.init_offset:
                self.curr_spike_idx += 1
                continue
            
            curr_channel = self.spike_events[self.curr_spike_idx][1]

            # Check if the next spike belongs to a channel that will already spike at the same time
            if curr_channel in spiking_channels:    # If the channel is already spiking, we stop the spikes for this time step
                break

            # Add the channel to the list of spiking channels
            spiking_channels.add(curr_channel)

            # Send a spike
            out_idx = self.channel_map[curr_channel]     # Map the channel to the output index
            if out_idx < self.s_out.shape[0]:   # Check if the channel is valid
                spike_data[out_idx] = 1.0   # Send a spike  (value corresponds to the punctual current of the spike event)

            # Move to the next spike event
            self.curr_spike_idx += 1

        # if len(spiking_channels) > 0:   # Print the spike event if there are any spikes
        #     print(f"sending spike event at time: {currTime}({self.time_step}) last spike idx: {self.curr_spike_idx-1} spike time: {self.spike_events[self.curr_spike_idx-1][0]}")

        # Send spikes
        # print("sending spike_data: ", spike_data, " at step: ", self.time_step)
        self.s_out.send(spike_data)

        # Stop the Process if there are no more spike events to send. (It will stop all the connected processes)
        # TODO: Should it be another process that stops the simulation? Such as the last LIF process
        # if self.curr_spike_idx >= 5: # len(self.spike_events):
        #    self.pause()





class SpikeEventGen_v2(AbstractProcess):
    """Input Process that generates spike events based on the input file
    This is the Version 2, where the input is given by a numpy array of spike events

    Args:
        @out_shape (tuple): Shape of the output port
        @spike_events (np.ndarray): Array of spike events of shape: (num_channels, num_events_per_ch)
        @name (str): Name of the process
    """
    def __init__(self, out_shape: tuple, spike_events: np.ndarray, name: str,
                 init_offset=0, virtual_time_step_interval=1) -> None:
        super().__init__(name=name)
        self.s_out = OutPort(shape=out_shape)

        self.spike_events = Var(shape=spike_events.shape, init=spike_events)
        spk_counters = np.zeros(self.spike_events.shape[0])    # Track the next spike event to send for each channel
        self.spk_counters = Var(shape=spk_counters.shape, init=spk_counters)

        self.init_offset = Var(shape=(1,), init=init_offset)
        self.virtual_time_step_interval = Var(shape=(1,), init=virtual_time_step_interval)

@implements(proc=SpikeEventGen_v2, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeEventGenModel_v2(PyLoihiProcessModel):
    """Spike Event Generator Process implementation running on CPU (Python) Version 2
    Args:
    """
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)   # IT IS POSSIBLE TO SEND FLOATS AFTER ALL

    '''
    spike_events contains the time steps and the channel of each spike event.
        Shape: (num_channels, num_events)
    '''
    spike_events: np.ndarray = LavaPyType(np.ndarray, object)
    spk_counters: np.ndarray = LavaPyType(np.ndarray, object)

    init_offset: int = LavaPyType(int, int)
    virtual_time_step_interval: int = LavaPyType(int, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params=proc_params)
        # print("spike events", self.spike_events.__str__())    # TODO: Check why during initialization the variable prints the class, while during run it prints the value

        # print("Spike Events: ", self.spike_events)
        # print("spk_counters: ", self.spk_counters)

    def run_spk(self) -> None:
        spike_data = np.zeros(self.s_out.shape) # Initialize the spike data to 0

        # print("time step:", self.time_step)

        # If the current simulation time is greater than a spike event, send a spike in the corresponding channel
        currTime = self.init_offset + self.time_step*self.virtual_time_step_interval

        num_input_channels = self.s_out.shape[0]    # Number of input channels
        for ch in range(num_input_channels):
            # Get the timestamp of the next spike event for the current channel
            next_spike_idx = int(self.spk_counters[ch])
            next_spike_time = -1
            if next_spike_idx < len(self.spike_events[ch]):
                # If the next spike event is valid, get its timestamp
                next_spike_time = self.spike_events[ch][next_spike_idx]

                if next_spike_time <= currTime:
                    # If the next spike event is less than the current time, the Input Layer should send a spike at this time step.
                    spike_data[ch] = 1.0   # Send spike (value corresponds to the punctual current of the spike event)

                    # Move to the next spike event
                    self.spk_counters[ch] += 1

        VERBOSE = False
        if VERBOSE and np.sum(spike_data) > 0:
            # Print the spike event if there are any spikes
            print(f"""Sending spike event at time: {currTime}({self.time_step}) | Data: {spike_data}""")
            #else:
            #     print(f"Sending spike event at time: {currTime}({self.time_step}).")

        self.s_out.send(spike_data)

        # Print a progress message every 1000 time steps
        if self.time_step % 1000 == 0:
            # Clear the console
            print(f"Time step: {self.time_step}")

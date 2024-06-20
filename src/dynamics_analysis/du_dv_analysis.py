import math
import numpy as np

def get_intersection_parameter_combinations(padding_time, max_spacing, sim_time, num_spikes, num_consec_spikes = None,
                                           consec_spikes_spacing = None, two_consec_spacing = None, v_th = 1.0, verbose = True):
    '''
    Get the intersection of the relevant results for the simulation of the CUBA LIF Neuron Dynamics, where:
    - Cond1: Spikes occur evenly spaced until max_spacing
    - Cond2: The last spike is delayed by 1 time step
    - Cond3: Spikes occur one right after the other (but with 1 spike less than num_spikes)

    Parameters:
    padding_time: Time to let the voltage settle to reach the maximum voltage
    max_spacing: The time interval between the first spike and the last spike of the burst
    sim_time: The total simulation time
    consec_spikes_spacing: The time interval between the spikes in the burst (for Cond3). If none, ignore Cond3
    num_spikes: The number of spikes in the burst
    v_th: The threshold voltage
    verbose: Whether to print the results
    '''
    CHECK_COND3 = consec_spikes_spacing is not None and num_consec_spikes is not None
    CHECK_COND4 = two_consec_spacing is not None

    # Fixed Parameters
    sim_time = padding_time + max_spacing    # 10ms simulation time (10 time steps)

    num_spikes = num_spikes                  # Number of spikes
    v_th = v_th                             # Threshold voltage

    # Calculate the spike times (make them evenly spaced until max_spacing)
    spike_times = [0]           # Let's consider the spikes are with maximum spacing
    for spike_idx in range(1, num_spikes):
        next_spike_time = (max_spacing / (num_spikes - 1)) * spike_idx
        # Convert the next_spike_time to an integer (round down)
        next_spike_time = int(next_spike_time)
        spike_times.append(next_spike_time)

    recording_times = [spike_times[-1]]         # The time step of interest is the step of the last stimuli before the burst

    # Explorable Parameters
    SPIKE_WEIGHTS = [0.05 * i for i in range(1, 20)]
    DU_VALS = [0.05 * i for i in range(1, 20)]
    DV_VALS = [0.05 * i for i in range(1, 20)]

    # Variables to store the results
    results_cond1 = []

    # Run the simulation for condition 1
    for spike_weight in SPIKE_WEIGHTS:
        for du in DU_VALS:
            for dv in DV_VALS:
                voltage_vals = voltage_pred(sim_time, du, dv, spike_times, recording_times, spike_weight)
                # print(f"bef_v: {bef_v}, v: {v}, after_v: {after_v}")

                bef_v, v, max_v = voltage_vals[0]

                currResult = CUBADynamicsResult(spike_weight, du, dv, bef_v, v, max_v)
                results_cond1.append(currResult)

    # Find the results where the Voltage is above the threshold (i.e. a spike occurs
    relevant_results1 = [result for result in results_cond1 if (
        result.bef_spike_v <= v_th 
        # and result.spike_v >= v_th 
        and result.max_spike_v >= v_th
        )]
    # Sort the relevant results from lowest voltage to highest
    relevant_results1 = sorted(relevant_results1, key=lambda item: item.max_spike_v, reverse=False)

    if verbose:
        print("Number of Results Cond 1: ", len(results_cond1))
        # print(results_cond1)

        # Print number of relevant results
        print(f"Number of relevant results for Cond. 1: {len(relevant_results1)}")
        # Show the relevant results
        print(relevant_results1)

        print("=====================================")

    # Let's now see condition 2 (delayed spike)
    # Fixed Parameters
    delayed_spike_times = spike_times          # Let's consider the spikes are with maximum spacing
    delayed_spike_times[-1] += 1    # Delay the last spike by 1 time step
    recording_times = [delayed_spike_times[-1]]

    # Variables to store the results
    results_cond2 = []

    # Run the simulation
    for spike_weight in SPIKE_WEIGHTS:
        for du in DU_VALS:
            for dv in DV_VALS:
                voltage_vals2 = voltage_pred(sim_time, du, dv, delayed_spike_times, recording_times, spike_weight)

                bef_v, v, max_v = voltage_vals2[0]
                # print(f"bef_v: {bef_v}, v: {v}, after_v: {after_v}")

                currResult = CUBADynamicsResult(spike_weight, du, dv, bef_v, v, max_v)
                results_cond2.append(currResult)

    # Find the results where the Voltage is below v_th (i.e. no spike occurs)
    relevant_results2 = [result for result in results_cond2 if (
        result.bef_spike_v < v_th and 
        result.spike_v < v_th and
        result.max_spike_v < v_th
        )]

    # Sort the relevant results from highest max voltage to lowest
    relevant_results2 = sorted(relevant_results2, key=lambda item: item.max_spike_v, reverse=True)

    if verbose:
        print("Number of Results Cond 2: ", len(results_cond2))

        # Print number of relevant results
        print(f"Number of relevant results Cond. 2: {len(relevant_results2)}")

        # Show the relevant results
        print(relevant_results2)

        print("=====================================")


    # Initialize the relevant results of cond. 3 as the relevant results of cond. 2 
    # in case cond. 3 is not checked
    relevant_results3 = relevant_results2   
    if CHECK_COND3:
        # Let's now see condition 3 (Consecutive Spikes)
        # Fixed Parameters
        num_consecutive_spikes = num_consec_spikes

        consecutive_spike_times = [spike_idx*consec_spikes_spacing for spike_idx in range(0, num_consecutive_spikes)]
            
        recording_times = [consecutive_spike_times[-1]]

        # Variables to store the results
        results_cond3 = []

        # Run the simulation
        for spike_weight in SPIKE_WEIGHTS:
            for du in DU_VALS:
                for dv in DV_VALS:
                    voltage_vals3 = voltage_pred(sim_time, du, dv, consecutive_spike_times, recording_times, spike_weight)

                    bef_v, v, max_v = voltage_vals3[0]
                    # print(f"bef_v: {bef_v}, v: {v}, after_v: {after_v}")

                    currResult = CUBADynamicsResult(spike_weight, du, dv, bef_v, v, max_v)
                    results_cond3.append(currResult)

        # Find the results where the Voltage is below v_th (i.e. no spike occurs)
        relevant_results3 = [result for result in results_cond3 if (
            result.bef_spike_v < v_th and 
            result.spike_v < v_th and
            result.max_spike_v < v_th
            )]

        # Sort the relevant results from highest max voltage to lowest
        relevant_results3 = sorted(relevant_results3, key=lambda item: item.max_spike_v, reverse=True)

        if verbose:
            print("Number of Results Cond 3: ", len(results_cond3))

            # Print number of relevant results
            print(f"Number of relevant results Cond. 3: {len(relevant_results3)}")

            # Show the relevant results
            print(relevant_results3)

            print("=====================================")

    relevant_results4 = relevant_results2   
    if CHECK_COND4:
        # Let's now see condition 4 (Two Consecutive Phases)
        # Fixed Parameters
        num_spikes_per_phase = num_spikes // 2

        two_phase_spike_times = []
        # Add the first phase spikes
        odd_increment = 1 if num_spikes % 2 == 1 else 0
        for spike_idx in range(0, num_spikes_per_phase + odd_increment):
            two_phase_spike_times.append(spike_idx*two_consec_spacing)
        # Add the second phase spikes
        space_between_phases = max_spacing // 2
        for spike_idx in range(0, num_spikes_per_phase):
            two_phase_spike_times.append(space_between_phases + spike_idx*two_consec_spacing)
        
        if verbose:
            print("Two Phase Spike Times: ", two_phase_spike_times)
            
        recording_times = [two_phase_spike_times[-1]]

        # Variables to store the results
        results_cond4 = []

        # Run the simulation
        for spike_weight in SPIKE_WEIGHTS:
            for du in DU_VALS:
                for dv in DV_VALS:
                    voltage_vals4 = voltage_pred(sim_time, du, dv, two_phase_spike_times, recording_times, spike_weight)

                    bef_v, v, max_v = voltage_vals4[0]
                    # print(f"bef_v: {bef_v}, v: {v}, after_v: {after_v}")

                    currResult = CUBADynamicsResult(spike_weight, du, dv, bef_v, v, max_v)
                    results_cond4.append(currResult)

        # Find the results where the Voltage is below v_th (i.e. no spike occurs)
        relevant_results4 = [result for result in results_cond4 if (
            result.bef_spike_v < v_th and 
            result.spike_v < v_th and
            result.max_spike_v < v_th
            )]

        # Sort the relevant results from highest max voltage to lowest
        relevant_results4 = sorted(relevant_results4, key=lambda item: item.max_spike_v, reverse=True)

        if verbose:
            print("Number of Results Cond 4: ", len(results_cond4))

            # Print number of relevant results
            print(f"Number of relevant results Cond. 4: {len(relevant_results4)}")

            # Show the relevant results
            print(relevant_results4)

            print("=====================================")


    # Get the intersection of the four sets
    relevant_results = []

    for rel_result in relevant_results1:
        if rel_result in relevant_results2 and rel_result in relevant_results3 and rel_result in relevant_results4:
            # Get the index of the relevant result
            index2 = relevant_results2.index(rel_result)
            index3 = relevant_results3.index(rel_result)
            index4 = relevant_results4.index(rel_result)
            delayed_result = relevant_results2[index2]
            consecutive_result = relevant_results3[index3]
            two_phase_result = relevant_results4[index4]

            relevant_results.append({
                "spike_weight": rel_result.spike_weight, "du": rel_result.du, "dv": rel_result.dv, 
                "max_v_on_time": rel_result.max_spike_v, "max_v_delayed": delayed_result.max_spike_v, 
                "max_v_consecutive": consecutive_result.max_spike_v, "max_v_two_phase": two_phase_result.max_spike_v,
                "spike_v_on_time": rel_result.spike_v, "bef_v_on_time": rel_result.bef_spike_v,
            })

            # print(f"Parameters: {rel_result}")
            # print(f"On Time Results: {rel_result.to_dict()}")
            # print(f"Voltage delayed: {relevant_results2[index].to_dict()}\n")

    # Sort the relevant results by lowest voltage later_v_on_time
    relevant_results = sorted(relevant_results, key=lambda item: item["max_v_on_time"], reverse=False)

    if verbose:
        print(f"Number of relevant results: {len(relevant_results)}")

        for rel_result in relevant_results:
            print(rel_result)

    return relevant_results



def voltage_pred(time, du, dv, spike_times, record_times,
                 spike_weight = 1.0, u_init = 0, v_init = 0):
    '''
    Calculates an approximation of the voltage of a CUBA LIF Neuron given the parameters at a given time step
    (Ignoring Spikes), i.e. only subthreshold dynamics are considered.
    @param time: the time step at which the voltage is to be calculated
    @param du: The inverse of the synaptic membrane time constant
    @param dv: The inverse of the membrane potential time constant
    @param record_times: The important times. The voltage will be recorded at
      each (t-1, t, max(t) before falling again)
    @param spike_times: The times at which the neuron spikes
    @param spike_weight The weight of the spikes

    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage

    @return: Array of the voltage at the recorded times
    '''
    curr_u = u_init
    curr_v = v_init
    recorded_times = []

    finding_v_max = False   # Flag to indicate if we are finding the max voltage after a spike
    curr_v_max = 0          # The current max voltage after a spike
    
    for time_step in range(0, time+1, 1):
        # print("Updating time step", time_step)
        spike_inc = spike_weight if time_step in spike_times else 0

        curr_u = curr_u * (1-du) + spike_inc
        curr_v = curr_v * (1-dv) + curr_u

        if time_step+1 in record_times:
            voltage_tuple = (curr_v, curr_v, curr_v)    # (v[t-1], v[t], v_max)
            recorded_times.append(voltage_tuple)
        elif time_step in record_times:
            recorded_times[-1] = (recorded_times[-1][0], curr_v, recorded_times[-1][2])      # Update the v(t) voltage
            finding_v_max = True
            curr_v_max = curr_v    # Set the current max voltage for this spike
        elif finding_v_max:
            if curr_v > curr_v_max:
                curr_v_max = curr_v
            elif curr_v <= curr_v_max:
                recorded_times[-1] = (recorded_times[-1][0], recorded_times[-1][1], curr_v_max)      # Update the v(t) voltage
                finding_v_max = False
                curr_v_max = 0
    
    return recorded_times


class CUBADynamicsResult:
    '''
    A class to represent an Result of the CUBA Dynamics Analysis
    '''
    def __init__(self, spike_weight, du, dv, bef_spike_v, spike_v, max_spike_v):
        self.spike_weight = spike_weight
        self.du = du
        self.dv = dv
        self.bef_spike_v = bef_spike_v
        self.spike_v = spike_v
        self.max_spike_v = max_spike_v

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
            "max v": self.max_spike_v
        }
    
    # Define method to print the object when printed
    def __repr__(self):
        return str(self.to_dict())
    
    # Define method to compare two objects
    def __eq__(self, other):
        return self.to_key() == other.to_key()

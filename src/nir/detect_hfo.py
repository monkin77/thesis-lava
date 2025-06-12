from utils.spike_event_gen import SpikeEventGen_v2
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from utils.data_analysis import find_spike_times
from nir_to_lava_edit import ImportConfig, LavaLibrary, import_from_nir
import nir
from utils.hfo import band_to_gt_max_offset, band_to_file_name, MarkerType, BaselineAlgorithm
from lava.proc.lif.process import LIFRefractory
from math import floor
import numpy as np
from utils.io import preview_np_array
from typing import TypedDict
from utils.float_to_fixed import scaling_dudv
from lava.magma.core.model.py.model import PyLoihiProcessModel


init_offset = 0 # 900 # 33400      #   
virtual_time_step_interval = 1  # dt = 1 ms

class HFOEvalResults(TypedDict):
    label: str
    TP: int
    FP: int
    FN: int
    TN: int
    recall: np.float32
    precision: np.float32
    f1_score: np.float32
    total_predictions: int

def float_to_fixed_process(key: str, proc: PyLoihiProcessModel, fp_scaling_factor=240):
    """
    Convert the process parameters to fixed-point representation.
    This is used for LIF and FC processes.
    """
    if "lif" in key:
        # Current Process is a LIF Process
        # Scale the dv and du parameters to fixed-point representation
        dv = scaling_dudv(proc.dv.get())    # .get() method needs to be used to get the value from the Lava Var)
        du = scaling_dudv(proc.du.get())
        vth = np.round(proc.vth.get() * fp_scaling_factor).astype(np.int32)
        print(f"Scaling LIF Process {key} to Fixed-Point: dv={dv}, du={du}, vth={vth}")
        # Update the Process with the new parameters
        proc.dv = dv
        proc.du = du
        proc.vth = vth
    elif "fc" in key:
        # Current Process is a Fully Connected Layer
        # Scale the Weights to fixed-point representation
        # print(f"Previous Weights: {proc.weights}")
        # .get() method is used to get the weights from the process since it is variable of class Lava Var
        weights = np.round(proc.weights.get() * fp_scaling_factor).astype(np.int32)
        print(f"Scaling FC Process {key} to Fixed-Point: weights={weights}")
        # Update the Process with the new weights
        proc.weights = weights

def evaluate_hfo_detector(
        nir_network: nir.NIRGraph, nir_config: ImportConfig, chosen_band: MarkerType, use_refrac: bool,
        input_spikes: np.ndarray, gt_times: np.ndarray, num_steps: int, use_fp: bool,
        eval_label: str, verbose=False) -> HFOEvalResults:
    '''
    Parameters
    ----------
    nir_network : nir.NIRGraph
        NIR Network to be imported
    nir_config : ImportConfig
        NIR Configuration to be used for the import
    chosen_band : MarkerType
        Band to be used for the evaluation (RIPPLE, FR)
    use_refrac : bool
        Whether to use the refractory period in the Output LIF Neurons
    input_spikes : np.ndarray
        Contains the UP and DOWN spikes that will be fed to the network. Shape: (2, num_spikes_per_ch)
    gt_times : np.ndarray
        Contains the GT times of the HFO events. Shape: (num_gt_events,)
    num_steps : int
        Number of time steps to run the network
    use_fp : bool
        Whether to use Fixed-Precision Computation in the network instead of Floating-Point
    eval_label : str
        Label to be used for the evaluation
    verbose : bool
        Whether to print verbose information
    '''
    # ========================================================================
    # Load the Lava Network from the NIR Network
    # ========================================================================
    lava_net, startNodes, endNodes = import_from_nir(
        nir_network,  # ng, #
        nir_config
    )
    if verbose:
        print(f"lava_net: {lava_net}")
        print(f"startNodes: {startNodes}")
        print(f"endNodes: {endNodes}")

        # Print the ports of the Network Processes
        for key, proc in lava_net.items():
            print(f"Key: {key} | Proc: {proc}")

            if isinstance(proc, list):
                # Network Node contains multiple Processes
                # Note: Max Depth is 2
                for inner_key, inner_proc in enumerate(proc):
                    print(f"Key: ({key},{inner_key}) | Proc: {inner_proc}")
                    # Print the Network Ports
                    for port in inner_proc.in_ports:
                        print(
                            f"Proc: {inner_proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")
                    for port in inner_proc.out_ports:
                        print(
                            f"Proc: {inner_proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")
            else:
                # Print the Network Ports
                for port in proc.in_ports:
                    print(
                        f"Proc: {proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")
                for port in proc.out_ports:
                    print(
                        f"Proc: {proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")

    # Get the Lava's Equivalent Start Process and Input and Output Ports
    # Define the Start Process of the Imported Network
    startProc = lava_net[startNodes[0]]

    # Test: Print the Ports of the Start Process
    in_port = startProc.a_in if hasattr(startProc, "a_in") else startProc.s_in
    out_port = startProc.a_out if hasattr(
        startProc, "a_out") else startProc.s_out
    if verbose:
        print("In Port: ", in_port, " | Size: ", in_port.size)
        print("Out Port: ", out_port, " | Size: ", out_port.size)

    # ========================================================================
    # Define Detection-Related Parameters
    # ========================================================================
    # Giving PRED_CAUSALITY_WINDOW ms for the network to update its inner state and spike
    PRED_CAUSALITY_WINDOW = int(5)
    # in timesteps (ms) - Max time from the Insertion Timing to the GT annotation
    MAX_DETECTION_OFFSET = int(band_to_gt_max_offset(
        chosen_band)) * 1.5 + PRED_CAUSALITY_WINDOW   # in timesteps (ms)
    if verbose:
        print(f"PRED_CAUSALITY_WINDOW: {PRED_CAUSALITY_WINDOW} steps")
        print(f"MAX_DETECTION_OFFSET: {MAX_DETECTION_OFFSET} ms")

    # ========================================================================
    # Make Lava-Specific Adjustments to the Network
    # Optional RefractoryLIF
    # ========================================================================
    # Add refractory period to lif_out neurons 
    lif_out = lava_net["lif_out"]
    REFRAC_PERIOD = 200     # floor(MAX_DETECTION_OFFSET) * 2    # Number of time-steps for the refractory period
    if verbose:
        print(f"Refractory Period: {REFRAC_PERIOD} steps")

    if use_refrac:
        # Calculate the dv and du from the time constants
        # Get the LifOut Parameters from the NIR Network
        nirLIFOut = nir_network.nodes["lif_out"]
        dv_lif_out = (nir_config.dt / nirLIFOut.tau_mem)
        du_lif_out = (nir_config.dt / nirLIFOut.tau_syn)
        if verbose:
            print(f"dv_lif_out: {dv_lif_out} | du_lif_out: {du_lif_out}")

        # Create the Refractory LIF Process
        lif_out_refrac = LIFRefractory(
            shape=(1,),  # There is 1 output neuron
            vth=nirLIFOut.v_threshold,
            dv=dv_lif_out,    # Inverse of decay time-constant for voltage decay
            du=du_lif_out,  # Inverse of decay time-constant for current decay
            refractory_period=REFRAC_PERIOD,
            name="cuba lif"
        )

        # Replace the original lif_out with the Refractory version
        lava_net["lif_out"] = lif_out_refrac
        # Make the port connections
        lava_net["fc_out"].a_out.connect(lif_out_refrac.a_in)
        lif_out = lif_out_refrac

    # ========================================================================
    # Transform the Network to use Fixed-Point Computation
    # ========================================================================
    if use_fp:
        '''
        Transform the Network to use Fixed-Point Computation
        For LIF Layers:
        - Convert the dv and du parameters to fixed-point representation
        - Convert the Voltage Threshold to fixed-point representation
        For Dense Layers:
        - Convert the Weights to fixed-point representation
        '''
        print(f"\nTransforming the Network to use Fixed-Point Computation...")
        FP_SCALING_FACTOR = 240  # Scaling factor for weights and v_th in Fixed-Point Representation
        # Iterate over the Lava Network and convert the parameters to fixed-point representation
        for key, proc in lava_net.items():
            # Allow 2 levels of depth in the Network. E.g.: Nested List of Processes [Dense, LIF]
            if isinstance(proc, list):
                # Network Node contains multiple Processes
                # Note: Max Depth is 2
                for inner_key, inner_proc in enumerate(proc):
                    float_to_fixed_process(inner_key, inner_proc, fp_scaling_factor=FP_SCALING_FACTOR)
            else:
                float_to_fixed_process(key, proc, fp_scaling_factor=FP_SCALING_FACTOR)

    if verbose:
        print(f"LAVA Network: {lava_net}\n\n")

    # ========================================================================
    # Create the Input Generator Layer and Connect it to the remaining Network
    # ========================================================================
    spike_event_gen = SpikeEventGen_v2(out_shape=(2,), spike_events=input_spikes, name="InputLayer",
                                       virtual_time_step_interval=virtual_time_step_interval, init_offset=init_offset)

    # Connect the Input Layer to the First Layer of the Trained HFO Detector
    # If I connect the SpikeEventGen to the Dense Layer, the a_out value of the custom input will be rounded to 0 or 1 in the Dense Layer (it will not be a float)
    spike_event_gen.s_out.connect(in_port)

    # ========================================================================
    # Run the Network
    # ========================================================================
    # Record Internal Variables
    # Get the Lava's LIF Objects corresponding to each of the LIF Nodes
    lif1 = lava_net["lif1"]  # [1]
    lif2 = lava_net["lif2"]  # [1]
    lif_out = lava_net["lif_out"]   # [1]

    # print("LIF1: ", lif1)
    # print("LIF2: ", lif2)
    # print("LIF_OUT: ", lif_out)

    monitor_lif_out_v = Monitor()
    monitor_lif_out_u = Monitor()
    monitor_lif_out_v.probe(lif_out.v, num_steps)
    monitor_lif_out_u.probe(lif_out.u, num_steps)

    # Define the Running Configuration and Conditions
    run_condition = RunSteps(num_steps=num_steps)
    # TODO: Check why we need this select_tag="floating_pt"
    run_cfg = Loihi1SimCfg(select_tag="floating_pt")

    # Run the Network
    spike_event_gen.run(condition=run_condition, run_cfg=run_cfg)

    # ========================================================================
    # Process the Network Output
    # ========================================================================
    # Merge the dictionaries to contain both voltage and current
    data_lif_out_v = monitor_lif_out_v.get_data()
    data_lif_out_u = monitor_lif_out_u.get_data()
    data_lif_out = data_lif_out_v.copy()
    data_lif_out["cuba lif"]["u"] = data_lif_out_u["cuba lif"]["u"]

    if verbose:
        print(
            f"Shape of LIF OUT Voltage Data: {data_lif_out_v['cuba lif']['v'].shape}")
        # print(data_lif_out_v)


    # Find the timesteps where the network spiked
    lif_out_voltage = np.array(data_lif_out['cuba lif']['v'])
    lif_out_current_vals = np.array(data_lif_out['cuba lif']['u'])

    # Call the find_spike_times util function that detects the spikes in a voltage array
    # TODO: Improve the find_spike_times method to view the current of the preview timestep to make sure it is a spike, instead of an inhibition
    spike_times_lif_out = find_spike_times(
        lif_out_voltage, lif_out_current_vals)

    if verbose:
        print(f"Found {len(spike_times_lif_out)} spikes in the Output LIF Process")
        for (spike_time, neuron_idx) in spike_times_lif_out:
            print(
                f"Spike time: {init_offset + spike_time * virtual_time_step_interval} (iter. {spike_time}) at neuron: {neuron_idx}")

    # ========================================================================
    # Compare the Network Predictions with the GT Annotations
    # ========================================================================
    # Preview the data
    if verbose:
        preview_np_array(gt_times, "gt_times", edge_items=2)

    # Get the Relevant Spike Times (Apply Refractory Period)
    real_spike_times = []
    prev_spike_anchor = -1
    for spike_time, _neuron_idx in spike_times_lif_out:
        if prev_spike_anchor < 0:
            real_spike_times.append(spike_time)
            prev_spike_anchor = spike_time
        else:
            # Check if the current spike time is within the range of the previous spike time
            if (spike_time - prev_spike_anchor) < REFRAC_PERIOD:
                # Skip this spike time
                continue
            else:
                real_spike_times.append(spike_time)
                prev_spike_anchor = spike_time

    if verbose:
        print(f"Found {len(real_spike_times)} real spikes in the Output LIF Process")
    real_spike_times = np.array(real_spike_times)

    # Change the number of values printed by numpy print
    if verbose:
        np.set_printoptions(threshold=1000)  # Set the threshold to a large value
        print(real_spike_times)

    # ========================================================================
    # Calculate Prediction Metrics
    # ========================================================================
    # We will consider a prediction correct if the network spikes in the interval `[gt_time, gt_time + MAX_DETECTION_OFFSET]`.
    TP, TN, FP, FN = 0, 0, 0, 0
    curr_gt_idx = 0
    for spike_time in real_spike_times:
        # Check if the spike time is within the range of the GT times
        while curr_gt_idx < len(gt_times):
            gt_time = gt_times[curr_gt_idx]
            if verbose:
                print(f"GT Time: {gt_time} | Spike Time: {spike_time}")
            if spike_time < gt_time:
                # This spike is before the insertion of a relevant event
                FP += 1
                break   # Exit the loop
            if (spike_time - gt_time) < MAX_DETECTION_OFFSET:
                # spike time must be >= gt_time
                TP += 1
                curr_gt_idx += 1
                break   # Exit the loop
            else:
                # The current spike is not relative to the next GT event
                # meaning that we did not predict the event
                FN += 1
                curr_gt_idx += 1
                continue   # Continue to the next GT event
        else:
            # No more GT events, so every spike is a false positive
            FP += 1

    # Show the Confusion Matrix
    print(f"Confusion Matrix:")
    # Print lines with same width
    print(f"|TP: {TP} | FP: {FP}|\n|FN: {FN}  | TN: {TN}|")

    # Calculate the performance metrics (Not including metrics that rely on TN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0
    total_predictions = TP + FP + FN
    # Output the performance metrics
    print(f"Recall (True Positive Rate): {recall*100:.2f} %")
    print(f"Precision (TP / (TP + FP)): {precision*100:.2f} %")
    print(f"F1 Score (Combines Precision & Recall): {f1_score*100:.2f} %")
    print(f"Total Predictions: {total_predictions}")
    # As we can see, the network is detecting HFOs with good Recall, Precision and F1-Score. Thus, we can conclude that the network
    # was successfully imported from `NIR` and is working in `LAVA`!

    # ========================================================================
    # Save the Results
    # ========================================================================
    ch_results: HFOEvalResults = {
        "label": eval_label,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "total_predictions": total_predictions
    }

    # ========================================================================
    # Clean Up The Network
    # ========================================================================
    # Terminates Process Execution for all Network Nodes
    spike_event_gen.stop()
    # Stop the monitors
    monitor_lif_out_v.stop()
    monitor_lif_out_u.stop()

    # Return the results 
    return ch_results

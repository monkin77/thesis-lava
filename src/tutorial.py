from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
import numpy as np

# Show the docstring of the LIF (Leaky-Integrate-and-Fire) Neural Process
# print(LIF.__doc__)

#############################################
#             Create processes              #
#############################################
lif1 = LIF(shape=(3,),                  # Nº and topological layout of units in the process
           vth=10.,                     # Membrane threshold 
           dv=0.1,                      # Inverse membrane time-constant
           du=0.1,                      # Inverse synaptic time-constant
           bias_mant=(1.1, 1.2, 1.3),   # Bias added to the membrane voltage in every timestep
           name="lif1"                  # Name of the process
           )

dense = Dense(weights=np.random.rand(2, 3), # Initial weights chosen randomly
              name="dense")

lif2 = LIF(shape=(2, ),             # Nº and topological layout of units in the process
           vth=10.,                 # Membrane threshold
           dv=0.1,                  # Inverse membrane time-constant
           du=0.1,                  # Inverse synaptic time-constant
           bias_mant=0.,            # Bias added to the membrane voltage in every timestep
           name="lif2"              # Name of the process
           )

#############################################
# See the size of the ports of each Process #
#############################################
for proc in [lif1, dense, lif2]:
    for port in proc.in_ports:
        print(f"Proc: {proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")
    for port in proc.out_ports:
        print(f"Proc: {proc.name:<5} Port Name: {port.name:<5} Size: {port.size}")

#############################################
#           Connect the processes           #
#############################################
lif1.s_out.connect(dense.s_in)  # Connect the spiking output port of lif1 to the spiking input port of dense
dense.a_out.connect(lif2.a_in)  # Connect the activation output port of dense to the activation input port of lif2

#############################################
#                 Variables                 #
#############################################
for var in lif1.vars:
    print(f"Var: {var.name:<5} Shape: {var.shape} Init: {var.init}")    # Print the name, shape and initial value of the variables of lif1

# Look at the weights of the Dense process by calling the get function
print(dense.weights.get())

#############################################
#       Record internal vars over time      #
#############################################
monitor_lif1 = Monitor()
monitor_lif2 = Monitor()
num_steps = 100

monitor_lif1.probe(lif1.v, num_steps)
monitor_lif2.probe(lif2.v, num_steps)

#############################################
#               Execution                   #
#############################################
# RunCondution defines how the network runs (i.e for how long)
run_continuous = RunContinuous()  # Run the processes continuously and non-blocking until pause or stop is called
run_steps = RunSteps(num_steps = num_steps)  # Run the processes for a fixed number of steps

# RunConfig defines on which hardware backend the Processes should be mapped and executed
run_cfg = Loihi1SimCfg(select_tag="floating_pt")    # Run on CPU -> Loihi1SimCfg

# Run the network
lif2.run(condition = run_steps, run_cfg = run_cfg)

#############################################
#           Retrieve recorded data          #
#############################################
data_lif1 = monitor_lif1.get_data()
data_lif2 = monitor_lif2.get_data()

#############################################
#           Plot recorded data              #
#############################################
import matplotlib.pyplot as plt

# Create a subplot for each monitor
fig = plt.figure(figsize=(16,5))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

# Plot the recorded data
monitor_lif1.plot(ax0, lif1.v)
monitor_lif2.plot(ax1, lif2.v)

#############################################
#           Stop the runtime                #
#############################################
lif2.stop()

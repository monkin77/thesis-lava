# Channel Burst detection


## Different Channel burst conditions

### 4 Spikes in 20ms range

In this case, using the lab data the first detected spikes are behaving correctly. However, when the spike frequency is very high, it takes less than 4 spikes to detect posterior spikes.

This could be solved by:
- Adding a refractory period (Time without spiking after a spike). 
  - This gives time to the neuron model to decrease the very high-current that is a consequence of the previous spike.
- Changing the dynamics?
  - I don't think this would help a lot. In my view, the simplicity of the LIF model makes it difficult to account for multiple network spikes. Adding the refractory period sounds like the better option.

#### Adding a refractory period
After adding the refractory period, the channel burst detection behaves as desired. The effect of previous spikes that already caused a channel burst is reduced, making the detection of new bursts more accurate.


## Time Interval
I found that the time interval between **5800ms** and **6100ms** has a lot of activity in multiple channels.



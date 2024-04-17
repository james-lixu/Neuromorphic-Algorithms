import numpy as np
import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
from nengo.dists import Choice
from nengo.utils.ensemble import tuning_curves
from nengo.processes import Piecewise

n_RL  = nengo.neurons.RectifiedLinear()
n_LIF = nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002)

J = np.linspace(-1,10,100)

plt.plot(J, n_RL.rates(J, gain=10, bias=0))
plt.xlabel('I')
plt.ylabel('$a$ (Hz)');
plt.show()

plt.plot(J, n_LIF.rates(J, gain=1, bias=0)) 
plt.xlabel('I')
plt.ylabel('a (Hz)'); 
plt.show()

model = nengo.Network(label='One Neuron')

with model:
    
    stimulus = nengo.Node(lambda t: np.sin(10*t))
    ens = nengo.Ensemble(1, dimensions=1, 
                         encoders = [[1]],
                         intercepts = [0.5],
                         max_rates= [100])
    
    nengo.Connection(stimulus, ens)
    
    spikes_p = nengo.Probe(ens.neurons, 'output')
    voltage_p = nengo.Probe(ens.neurons, 'voltage')
    stim_p = nengo.Probe(stimulus)

sim = nengo.Simulator(model)
sim.run(1)

t = sim.trange()     
plt.figure(figsize=(10,4))
plt.plot(t, sim.data[stim_p], label='stimulus', color='r', linewidth=4)
plt.ax = plt.gca()
plt.ax.plot(t, sim.data[voltage_p],'g', label='v')
plt.ylim((-1,2))
plt.ylabel('Voltage')
plt.xlabel("Time")
plt.legend(loc='lower left')

rasterplot(t, sim.data[spikes_p], ax=plt.ax.twinx(), use_eventplot=True)
plt.ylim((-1,2))
plt.show()


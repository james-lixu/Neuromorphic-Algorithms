import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.processes import WhiteNoise
import numpy as np
import matplotlib.pyplot as plt

#Transforming sin(x) to 2sin(x) by decoder scaling
T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim = nengo.Node(lambda t: 0.5*np.sin(10*t))
    ensA = nengo.Ensemble(100, dimensions=1)
    ensB = nengo.Ensemble(100, dimensions=1)
    
    nengo.Connection(stim, ensA)
    nengo.Connection(ensA, ensB, transform=2) #function=lambda x: 2*x)
    
    stim_p = nengo.Probe(stim)
    ensA_p = nengo.Probe(ensA, synapse=.01)
    ensB_p = nengo.Probe(ensB, synapse=.01)
    ensA_spikes_p = nengo.Probe(ensA.neurons, 'output')
    ensB_spikes_p = nengo.Probe(ensB.neurons, 'output')
   
sim = nengo.Simulator(model, seed=4)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6, 4))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p],'r', linewidth=4, label='x')
plt.plot(t, sim.data[ensA_p],'g', label='$\hat{x}$')
plt.plot(t, sim.data[ensB_p],'b', label='$f(\hat{x})=2\hat{x}$')
plt.legend()
plt.ylabel("Output")
plt.xlabel("Time")
plt.show()

#Transforming sin(x) to sin(x)^2 
T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stim = nengo.Node(lambda t: 0.5*np.sin(10*t))
    ensA = nengo.Ensemble(100, dimensions=1)
    ensB = nengo.Ensemble(100, dimensions=1)
    
    nengo.Connection(stim, ensA)
    nengo.Connection(ensA, ensB, function=lambda x: x**2)
    
    stim_p = nengo.Probe(stim)
    ensA_p = nengo.Probe(ensA, synapse=.01)
    ensB_p = nengo.Probe(ensB, synapse=.01)
    ensA_spikes_p = nengo.Probe(ensA.neurons, 'output')
    ensB_spikes_p = nengo.Probe(ensB.neurons, 'output')
   
sim = nengo.Simulator(model, seed=4)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6, 4))
plt.ax = plt.gca()
plt.plot(t, sim.data[stim_p],'r', linewidth=4, label='x')
plt.plot(t, sim.data[ensA_p],'g', label='$\hat{x}$')
plt.plot(t, sim.data[ensB_p],'b', label='$f(\hat{x})=\hat{x}^2$')
plt.legend()
plt.ylabel("Output")
plt.xlabel("Time")
plt.show()

# Summation of two shifted sin functions
T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stimA = nengo.Node(lambda t: 0.5*np.sin(10*t))
    stimB = nengo.Node(lambda t: 0.5*np.sin(5*t))
    
    ensA = nengo.Ensemble(100, dimensions=1)
    ensB = nengo.Ensemble(100, dimensions=1)
    ensC = nengo.Ensemble(100, dimensions=1)
    
    nengo.Connection(stimA, ensA)
    nengo.Connection(stimB, ensB)
    nengo.Connection(ensA, ensC)
    nengo.Connection(ensB, ensC)
    
    stimA_p = nengo.Probe(stimA)
    stimB_p = nengo.Probe(stimB)
    ensA_p = nengo.Probe(ensA, synapse=.01)
    ensB_p = nengo.Probe(ensB, synapse=.01)
    ensC_p = nengo.Probe(ensC, synapse=.01)
   
sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure(figsize=(6,4))
plt.plot(t, sim.data[ensA_p],'b', label="$\hat{x}$")
plt.plot(t, sim.data[ensB_p],'m--', label="$\hat{y}$")
plt.plot(t, sim.data[ensC_p],'k--', label="$\hat{x}+\hat{y}$")
plt.legend(loc='best')
plt.ylabel("Output")
plt.xlabel("Time")
plt.show()

#Summation of two encoded vectors
T = 1
max_freq = 5

model = nengo.Network()

with model:

    stimA = nengo.Node([.3,.5])
    stimB = nengo.Node([.3,-.5])
    
    ensA = nengo.Ensemble(100, dimensions=2)
    ensB = nengo.Ensemble(100, dimensions=2)
    ensC = nengo.Ensemble(100, dimensions=2)
    
    nengo.Connection(stimA, ensA)
    nengo.Connection(stimB, ensB)
    nengo.Connection(ensA, ensC)
    nengo.Connection(ensB, ensC)
    
    stimA_p = nengo.Probe(stimA)
    stimB_p = nengo.Probe(stimB)
    ensA_p = nengo.Probe(ensA, synapse=.02)
    ensB_p = nengo.Probe(ensB, synapse=.02)
    ensC_p = nengo.Probe(ensC, synapse=.02)
   
sim = nengo.Simulator(model)
sim.run(T)

plt.figure()
plt.plot(sim.data[ensA_p][:,0], sim.data[ensA_p][:,1], 'g', label="$\hat{x}$")
plt.plot(sim.data[ensB_p][:,0], sim.data[ensB_p][:,1], 'm', label="$\hat{y}$")
plt.plot(sim.data[ensC_p][:,0], sim.data[ensC_p][:,1], 'k', label="$\hat{x} + \hat{y}$")
plt.ylabel("$x_2$")
plt.xlabel("$x_1$")
plt.legend(loc='best')
plt.show()

#Multiplication of two 2D vectors
T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stimA = nengo.Node(lambda t: 0.5*np.sin(10*t))
    stimB = nengo.Node(lambda t: 0.5*np.sin(5*t))
    
    ensA = nengo.Ensemble(200, dimensions=2)
    ensB = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(stimA, ensA[0])
    nengo.Connection(stimB, ensA[1])

    nengo.Connection(ensA, ensB, function=lambda x: x[0]*x[1])
    
    stimA_p = nengo.Probe(stimA)
    stimB_p = nengo.Probe(stimB)
    ensA_p = nengo.Probe(ensA, synapse=.01)
    ensB_p = nengo.Probe(ensB, synapse=.01)
   
sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure()
plt.plot(t, sim.data[ensA_p][:,0],'black', label="$\hat{x}[0]$")
plt.plot(t, sim.data[ensA_p][:,1],'black', label="$\hat{x}[1]$")
plt.plot(t, sim.data[ensB_p],'r', label="$\hat{x[0]}\cdot\hat{x[1]}$")
plt.legend(loc='best')
plt.ylabel("Output")
plt.xlabel("Time")
plt.show()

#Signal Getting 
T = 1.0
max_freq = 5

model = nengo.Network()

with model:
    stimA = nengo.Node(lambda t: 0.5*np.sin(10*t))
    stimB = nengo.Node(lambda t: 0 if (t<.5) else 1)
    
    ensA = nengo.Ensemble(300, dimensions=2, radius=np.sqrt(2))
    ensB = nengo.Ensemble(100, dimensions=1)
    nengo.Connection(stimA, ensA[0])
    nengo.Connection(stimB, ensA[1])
    nengo.Connection(ensA, ensB, function=lambda x: x[0]*x[1])
    
    stimA_p = nengo.Probe(stimA)
    stimB_p = nengo.Probe(stimB)
    ensA_p = nengo.Probe(ensA, synapse=.01)
    ensB_p = nengo.Probe(ensB, synapse=.01)
   
sim = nengo.Simulator(model)
sim.run(T)

t = sim.trange()
plt.figure()
plt.plot(t, sim.data[ensA_p][:,0],'black', label="$\hat{x}[0]$")
plt.plot(t, sim.data[ensA_p][:,1],'blue', label="$\hat{x}[1]$")
plt.plot(t, sim.data[ensB_p],'r', label="$\hat{x[0]}\cdot\hat{x[1]}$")
plt.legend(loc='best')
plt.ylabel("Output")
plt.xlabel("Time")
plt.show()

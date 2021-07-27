
"""
Created on Mar  15 14:34:00 2019

@author: rodrigo pena
"""

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

##########################################################################################
# Parameters
##########################################################################################
raster=0
cross=1
trials=100

dt = 0.1 * ms
defaultclock.dt = dt
seed_per=1000

NE = 2
gamma = 0.25
NI = 2 #int(gamma*NE)
CE = 1
CI = 1 #int(gamma*CE)

simulation_time = 2000 * ms
transient = 0*ms

#model1 parameters
El=-65
Ena = 55
Eh = -20
Gl=0.5
Gp = 0.5
# Gh = 1.5 #3.5 #1.5
Vhlf_p = -38.0
Vslp_p = 6.5
Vhlf_h = -79.2
Vslp_h = 9.78
vth = -45
vr  = -80
rrst_e = 0
u=-2.3
# vary from 30-190
rtaue = 40*ms #80*ms #40*ms #160*ms #80*ms
tau_I=0.04*ms #0.04*ms

#model2 parameters
# El=-75.
# Ena = 42.
# Eh = -26.
# u=-0.6
# Gl=0.3
# Gp = 0.08
# Gh = 1.5
# Vhlf_p = -54.8
# Vslp_p = 4.4
# Vhlf_h = -74.2
# Vslp_h = 7.2
# vth = -51.
# vr  = -75.
# rrst_e = 0.0
# J = 0.3 #5.0 #0.3 

#syn parameters
J=6.0 #3.0
g=4.5


def fixed_indegree(indegree,n_post_pop,n_pre_pop,seed):
    np.random.seed(seed)
    presyn_indices = np.zeros([n_post_pop*indegree])
    postsyn_indices = np.zeros([n_post_pop*indegree])
    counter = 0

    for post in range(n_post_pop):
        x = np.arange(0, n_pre_pop)
        y = np.random.permutation(x)
        for i in range(indegree):
            presyn_indices[counter] = y[i]
            postsyn_indices[counter] = post
            counter += 1
    presyn_indices = presyn_indices.astype(int)
    postsyn_indices = postsyn_indices.astype(int)
    return presyn_indices, postsyn_indices

presyn_indices,postsyn_indices=fixed_indegree(CE,(NE+NI),NE,seed_per)
presyn_indices,postsyn_indices=fixed_indegree(CI,(NE+NI),NI,seed_per)

#reset seed for simulation
seed()
np.random.seed()

##########################################################################################
# Neuron
##########################################################################################

eqs1 = '''
        pinf = 1.0/(1+exp(-(v-Vhlf_p)/Vslp_p)) : 1
        rinfe= 1.0/(1+exp((v-Vhlf_h)/Vslp_h)) : 1
        dv/dt = (-Gl*(v - El) + u - Gp*pinf*(v-Ena) - Gh*r*(v-Eh) + sqrt(2*tau_I)*xi) / ms  : 1
        dr/dt =  (rinfe-r)/rtaue : 1
        Gh : 1
        ''' 

reset_eqs='''
        v=vr
        r=rrst_e
        '''

neurons = NeuronGroup(NE+NI, eqs1, method='euler', dt=dt,
                             threshold='v>=vth', reset=reset_eqs)
pop_e = neurons[:NE]
pop_i = neurons[NE:(NE+NI)]                             

#ICs
neurons.v = -60
neurons.r = 0

pop_e.Gh=1.5
pop_i.Gh=3.5

##########################################################################################
#  Connections
##########################################################################################
#E indexes
con_e = Synapses(pop_e, neurons, on_pre='v += J',dt=dt,delay=1.5*ms)
con_e.connect(i=presyn_indices,j=postsyn_indices)  
    
#I indexes
con_i = Synapses(pop_i, neurons, on_pre='v += J',dt=dt,delay=1.5*ms) #on_pre='v -= J*g'
con_i.connect(i=presyn_indices,j=postsyn_indices)                 


##########################################################################################
# Input load
##########################################################################################

# x = pd.read_csv('pre_corrected_if.dat', header=None,delimiter='   ')
# inj = []
# values_syn = []
# for i in range(Nneurons):
#     ind = range(len(np.asarray(x[i])))
#     times = np.nonzero((ind*np.asarray(x[i])))[0]*dt
#     indices = np.zeros((1, len(times)))
#     inj_spikes = SpikeGeneratorGroup(1, indices[0], times)
#     inj.append(inj_spikes)
#     values_syn.append(Synapses(inj[i], quadra_neurons[i:i+1], on_pre='v += J', delay=1.5*ms))

# print('done creating synapses')
##########################################################################################
# Connecting
##########################################################################################
# keys_synapses = range(len(values_syn))
# syn_dict = dict(zip(keys_synapses, values_syn))

# for i in range(len(syn_dict)):
#         syn_dict[i].connect()

# print('done connecting synapses')
##########################################################################################
# Running/recording
##########################################################################################

ratemon = PopulationRateMonitor(neurons)
spkmon = SpikeMonitor(neurons)

# rec_v = StateMonitor(quadra_neurons,'v',record=True)
# rec_r = StateMonitor(quadra_neurons,'r',record=True)


net = Network(neurons, con_e, con_i, spkmon, ratemon) #, rec_v, rec_r

cxy = np.zeros((NE+NI,NE+NI,39999))

net.store()
for kk in range(trials):
        print(str(kk))
        net.run(simulation_time)

        if(raster):
                plt.subplot(2,1,1)
                plt.plot(spkmon.t/ms,spkmon.i,'k.',markersize=0.5)
                # ylim(950,1000)
                
                plt.subplot(2,1,2)
                plt.plot(ratemon.t/ms, ratemon.rate/Hz)
                plt.show()

        ##########################################################################################
        # Data manipulation
        ##########################################################################################
        '''convert spike-train'''

        spks = spkmon.spike_trains()      
        spike_train = np.zeros((NE+NI, int((simulation_time/ms)/(dt/ms))))  # to be recorded
        for i in range(NE+NI):
                x = np.zeros(int((simulation_time/ms)/(dt/ms)))
                a = ((spks[i])/ms)/(dt/ms) - transient/dt  # /(defaultclock.dt//ms)
                x[a.astype(int)] = 1/(dt/ms)
                spike_train[i, :] = x

        # np.savetxt('spk_train.dat',spike_train.T) 


        ''' compute cross-correlations '''
        if(cross):
                lags = np.arange(-(simulation_time/ms) ,(simulation_time/ms)-dt/ms,dt/ms)
                for i in range(NE+NI):
                        for j in range(NE+NI):
                                x = spike_train[i,:]
                                y = spike_train[j,:]
                                cxy[i,j,:] = cxy[i,j,:] + signal.fftconvolve(x, y[::-1], mode='full') 
                                # plt.subplot(12,12,i + j*i)
                                # plt.figure()
                                # plt.plot(lags,cxy)
                                # plt.xlim([-20,20])
                                # plt.show()
        net.restore()
cxy = np.divide(cxy,trials)

# net.store()
#         for kk in range()
#         net.run(simulation_time, report='text')

#         spks = spike_quadra.spike_trains()
#         spike_train = np.zeros(
#         (Nneurons, int((simulation_time/ms)/(dt/ms))))  # to be recorded
#         for i in range(Nneurons):
#                 x = np.zeros(int((simulation_time/ms)/(dt/ms)))
#                 a = ((spks[i])/ms)/(dt/ms) - transient/dt  # /(defaultclock.dt//ms)
#                 x[a.astype(int)] = 1/(dt/ms)
#                 spike_train[i, :] = x

#         np.savetxt('post_model1_tauh' + str(kk) + '.dat',spike_train.T)
#         net.restore()
# y = spks.t/ms
# x = spks.i
# a = numpy.array([x,y])
# numpy.savetxt('raster_g3_vex_vth25_D2_AR.dat',a.transpose())

# plt.plot(lags,cxy[4,0,:])
# plt.xlim([-20,20])
# plt.show()

# for i in range(12):
#         for j in range(12):
#                 plt.subplot(12,12,(i+j*12+1))
#                 plt.plot(lags,cxy[i,j,:])
#                 plt.xlim([-40,40])
# plt.show()

for i in range(4):
        for j in range(4):
                plt.subplot(4,4,(i+j*4+1))
                plt.plot(lags,cxy[i,j,:])
                plt.xlim([-20,20])
plt.show()

"""
Created on Mar  15 14:34:00 2019

@author: rodrigo pena
"""

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pylab
pylab.rcParams['savefig.dpi'] = 120
from detect_peaks import *
from scipy.signal import savgol_filter   

##########################################################################################
# Parameters
##########################################################################################
raster=0
cross=1
connmtx=0
trials=100

dt = 0.1 * ms
defaultclock.dt = dt
seed_per=8300 

NE = 2
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

connM = np.zeros((NE+NI,NE+NI))
presyn_indices,postsyn_indices=fixed_indegree(CE,(NE+NI),(NE+NI),seed_per*2)
for i in range(len(presyn_indices)):
        connM[presyn_indices[i],postsyn_indices[i]]=1


'''Connectivity figure'''
if(connmtx):
        plt.figure(figsize=(10,8))
        im=plt.imshow(connM,origin='upper',aspect='equal')
        ax=plt.gca()
        # Major ticks
        ax.set_xticks(np.arange(0, NE+NI, 1))
        ax.set_yticks(np.arange(0, NE+NI, 1))
        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, NE+NI+1, 1))
        ax.set_yticklabels(np.arange(1, NE+NI+1, 1))
        # # Minor ticks
        ax.set_xticks(np.arange(-.5, 3.5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3.5, 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=4)
        plt.xlabel('postsynaptic')
        plt.ylabel('presynaptic')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
        plt.tight_layout()              
        # plt.savefig('adj.png')
        # plt.savefig('adj.eps')   
        plt.show()

#reset seed for simulation
seed()
np.random.seed()

##########################################################################################
# Neuron model
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
#indexes
con_e = Synapses(neurons, neurons, on_pre='v += J',dt=dt,delay=1.5*ms)
con_e.connect(i=presyn_indices,j=postsyn_indices)  
    
##########################################################################################
# Running/recording
##########################################################################################

ratemon = PopulationRateMonitor(neurons)
spkmon = SpikeMonitor(neurons)

net = Network(neurons, con_e, spkmon, ratemon) #, rec_v, rec_r

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
                                x = spike_train[j,:]
                                y = spike_train[i,:]
                                cxy[i,j,:] = cxy[i,j,:] + signal.fftconvolve(x, y[::-1], mode='full') 
                                # plt.subplot(12,12,i + j*i)
                                # plt.figure()
                                # plt.plot(lags,cxy)
                                # plt.xlim([-20,20])
                                # plt.show()
        net.restore()
cxy = np.divide(cxy,trials)

if(cross):
        plt.figure(figsize=(10,8))
        for i in range(4):
                for j in range(4):
                        ax=plt.subplot(4,4,(i+j*4+1)) #
                        plt.plot(lags,cxy[j,i,:])
                        plt.xlim([-10,10])
                        # plt.ylim([0,300])
                        if(i==0):
                                plt.ylabel(r'$c_{xy}(\tau)$')
                        if(j==3):
                                plt.xlabel(r'$\tau$ [ms]')  
                        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(20)
        plt.tight_layout()              
        # plt.savefig('cross.png')
        # plt.savefig('cross.eps')             
        plt.show()

plt.figure(figsize=(10,8))
for i in range(4):
        for j in range(4):
                ax=plt.subplot(4,4,(i+j*4+1))
                yhat = savgol_filter(cxy[j,i,:], 9, 3)
                ids=detect_peaks(yhat,mpd=25,mph=50)
                ids=ids[ids>20000]
                ids=ids[ids<20100]
                plt.plot(lags[ids],yhat[ids],'x')
                plt.plot(lags,yhat)
                plt.xlim([-10,10])
                print(lags[ids])
                print(yhat[ids])
plt.show()
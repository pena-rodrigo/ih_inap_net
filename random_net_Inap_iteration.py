
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
sim_num=1

dt = 0.1 * ms
defaultclock.dt = dt

NE = 2
NI = 2 #int(gamma*NE)
CE = 1
CI = 1 #int(gamma*CE)

simulation_time = 2000 * ms
transient = 0*ms

#model1 parameters
# El1=-65
# Ena1 = 55
# Eh1 = -20
# Gl1=0.5
# Vhlf_p1 = -38.0
# Vslp_p1 = 6.5
# Vhlf_h1 = -79.2
# Vslp_h1 = 9.78
# vth1 = -45
# vr1  = -80
# rrst_e1 = 0
# u1=-2.3
# # vary from 30-190
rtaue1 = 40*ms #80*ms #40*ms #160*ms #80*ms
tau_I=0.04*ms #0.04*ms

#model2 parameters
El2=-75.
Ena2 = 42.
Eh2 = -26.
u2=-0.6
Gl2=0.3
Gp2 = 0.08
Gh2 = 1.5
Vhlf_p2 = -54.8
Vslp_p2 = 4.4
Vhlf_h2 = -74.2
Vslp_h2 = 7.2
vth2 = -51.
vr2  = -75.

#syn parameters
J=6.0 #6.0 #3.0
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

case1_lags = []
case1_height = [] 
case2_lags1 = []
case2_height1 = []
case2_lags2 = []
case2_height2 = [] 
positive = 0
negative = 0
numcons = 0

crss = []
lbs = []
for seed_per in range(sim_num):
# seed_per=8300     

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
                pinf = 1.0/(1+exp(-(v-Vhlf_p1)/Vslp_p1)) : 1
                rinfe= 1.0/(1+exp((v-Vhlf_h1)/Vslp_h1)) : 1
                dv/dt = (-Gl1*(v - El1) + u1 - Gp*pinf*(v-Ena1) - Gh*r*(v-Eh1) + sqrt(2*tau_I)*xi) / ms  : 1
                dr/dt =  (rinfe-r)/rtaue1 : 1
                Gh : 1
                Gp : 1
                El1 : 1
                Ena1 : 1
                Eh1 : 1
                Gl1 : 1
                Vhlf_p1 : 1
                Vslp_p1 : 1
                Vhlf_h1 : 1
                Vslp_h1 : 1
                vth1 : 1
                vr1  : 1
                rrst_e1 : 1
                u1 : 1
                ''' 

        reset_eqs1='''
                v=vr1
                r=rrst_e1
                '''

        neurons = NeuronGroup(NE+NI, eqs1, method='euler', dt=dt,
                                threshold='v>=vth1', reset=reset_eqs1)
        pop_e = neurons[:NE]
        pop_i = neurons[NE:(NE+NI)]                             

        #ICs
        neurons.v = -60
        neurons.r = 0

        pop_e.Gh=1.5
        pop_e.Gp = 0.5
        pop_e.El1=-65
        pop_e.Ena1 = 55
        pop_e.Eh1 = -20
        pop_e.Gl1=0.5
        pop_e.Vhlf_p1 = -38.0
        pop_e.Vslp_p1 = 6.5
        pop_e.Vhlf_h1 = -79.2
        pop_e.Vslp_h1 = 9.78
        pop_e.vth1 = -45
        pop_e.vr1  = -80
        pop_e.rrst_e1 = 0
        pop_e.u1=-2.3

        pop_i.Gh=1.5
        pop_i.Gp = 0.08
        pop_i.El1=-75.
        pop_i.Ena1 = 42.
        pop_i.Eh1 = -26.
        pop_i.Gl1=0.3
        pop_i.Vhlf_p1 = -54.8
        pop_i.Vslp_p1 = 4.4
        pop_i.Vhlf_h1 = -74.2
        pop_i.Vslp_h1 = 7.2
        pop_i.vth1 = -51.
        pop_i.vr1  = -75.
        pop_i.rrst_e1 = 0
        pop_i.u1=-0.6

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

        for i in range(NE*NI):
                for j in range(NE*NI):
                        yhat = savgol_filter(cxy[j,i,20000:20100], 9, 3)
                        if(i in postsyn_indices[where(presyn_indices==j)] and i > 1):
                                label = 1 ## Gh=1.5
                        elif(i in postsyn_indices[where(presyn_indices==j)] and i <= 1):
                                label = 2 ## Gh=3.5
                        else:
                                label = 0
                        crss.append(yhat)
                        lbs.append(label)

# np.savetxt('crss.dat',crss)
# np.savetxt('lbs.dat',lbs)
case1_lags = []
case1_height = [] 
case2_lags = []
case2_height = [] 
positive = 0
negative = 0
plt.figure(figsize=(10,8))
for i in range(4):
        for j in range(4):
                ax=plt.subplot(4,4,(i+j*4+1))
                yhat = cxy[j,i,:]#savgol_filter(cxy[j,i,:], 9, 3)
                ids=detect_peaks(yhat,mpd=25,mph=50)
                ids=ids[ids>20000]
                ids=ids[ids<20100]
                plt.plot(lags[ids],yhat[ids],'x')
                plt.plot(lags,yhat)
                plt.xlim([-10,10])
                print(lags[ids])
                print(yhat[ids])
                if(i in postsyn_indices[where(presyn_indices==j)] and i > 1):
                        if(len(ids)==1):
                                positive += 1
                        else:
                                negative +=1
                                print(i)
                                print(j)
                elif(i in postsyn_indices[where(presyn_indices==j)] and i <= 1):
                        if(len(ids)==2):
                                positive += 1
                        else:
                                negative +=1
                                print(i)
                                print(j)                           
                if(len(ids)==1):
                        case1_lags.append(lags[ids].item())
                        case1_height.append(yhat[ids].item()) 
                elif(len(ids)==2):
                        case2_lags.append(lags[ids][0])
                        case2_height.append(yhat[ids][0]) 
                        case2_lags.append(lags[ids][1])
                        case2_height.append(yhat[ids][1]) 

plt.show()
import numpy as np
from scipy import signal
import sys
import os

# Gl=float(sys.argv[-1])
tauh=80

preFileName = 'pre_if.dat'
# postFileName = 'post_model1_tauh' + str(tauh) + '.dat'
# postFileName = 'post_model1_tauh80_below51.dat'
postFileName = 'post_model2_gh25.dat'


x = np.loadtxt(preFileName).T
y = np.loadtxt(postFileName).T
time = 2000
ncells = 300
dt=0.1

cxy = np.zeros(39999)
for i in range(ncells):
    spk_train1 = x[i]
    spk_train2 = y[i]
    cxy = cxy + signal.fftconvolve(spk_train2, spk_train1[::-1], mode='full') 

cxy = np.divide(cxy,ncells)
lags = np.arange(-time,time-dt,dt)

results = np.array((lags,cxy)).T
# np.savetxt('cxy_model1_tauh' + str(tauh) + 'below51.dat',results)
# np.savetxt('cxy_model1_tauh' + str(tauh) + '.dat',results)
np.savetxt('cxy_model2_gh25.dat',results)


cxx = np.zeros(39999)
for i in range(ncells-1):
    spk_train1 = y[i+1]
    spk_train2 = y[i]
    cxx = cxx + signal.fftconvolve(spk_train2, spk_train1[::-1], mode='full') 

cxx = np.divide(cxx,ncells-1)
lags = np.arange(-time,time-dt,dt)

results = np.array((lags,cxx)).T
# np.savetxt('cxx_model1_tauh' + str(tauh) + 'below51.dat',results)
# np.savetxt('cxx_model1_tauh' + str(tauh) + '.dat',results)
np.savetxt('cxx_model2_gh25.dat',results)


# remove = 'rm ' + postFileName
# os.system(remove)
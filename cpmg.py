import mripy as mp
import numpy as np
import matplotlib.pyplot as plt
import time as tim

startTime = tim.perf_counter()
#basic simulation of a 10 pulse cpmg pulsesequence
T1 = 820000
T2 = 710000
isochromats = 101
TE = 5e3
acquisition_time = 2000
DW = 10
ETL = 200

#create initial magnetization vectors with lorentzian distribution
offset = np.linspace(-500,500,isochromats)
M=np.zeros([isochromats,3,1])
M[:,2,0]=mp.lorentzian(offset,a=1,c=100)

#create time evolution matrix for 1us (precession from offset, T1/T2 relaxation)
(AA,BB)=mp.relaxation(1,T1,T2);
rotMatrix_RF = np.zeros((isochromats,3,3))
for i in range(isochromats):
    rotMatrix_RF[i,:,:]=mp.rotZ(offset[i]*2*np.pi/10**6)
rotMatrix_RF = np.matmul(rotMatrix_RF,AA)
BB_RF = M*BB

#create time evolution matrix for TE/2
(AA,BB)=mp.relaxation(TE/2,T1,T2);
rotMatrix_TE = np.zeros((isochromats,3,3))
for i in range(isochromats):
    rotMatrix_TE[i,:,:]=mp.rotZ(offset[i]*2*np.pi/10**6*TE/2)
rotMatrix_TE = np.matmul(rotMatrix_TE,AA)
BB_TE = M*BB

#create time evolution matrix for TE/2 - acquisition/2
(AA,BB)=mp.relaxation(TE/2-acquisition_time,T1,T2);
rotMatrix_acq = np.zeros((isochromats,3,3))
for i in range(isochromats):
    rotMatrix_acq[i,:,:]=mp.rotZ(offset[i]*2*np.pi/10**6*(TE/2-acquisition_time))
rotMatrix_acq = np.matmul(rotMatrix_acq,AA)
BB_acq = M*BB

#create time evolution matrix for DW
(AA,BB)=mp.relaxation(DW,T1,T2);
rotMatrix_DW = np.zeros((isochromats,3,3))
for i in range(isochromats):
    rotMatrix_DW[i,:,:]=mp.rotZ(offset[i]*2*np.pi/10**6*DW)
rotMatrix_DW = np.matmul(rotMatrix_DW,AA)
BB_DW = M*BB

#prealocate the xy-magnetization vector for 100000 timesteps
XY = np.zeros(ETL*(acquisition_time//DW))
time = 0

#perform simulation

#pi/2 pulse
(M,_) = mp.rectPulse(np.pi/2, M, rotMatrix_RF, BB_RF, phi=0,t=100)
# time evolution until pi pulse
M = np.matmul(rotMatrix_TE,M)+BB_TE

for i in range(ETL):
    #pi pulse
    (M,_) = mp.rectPulse(np.pi, M, rotMatrix_RF, BB_RF, phi=np.pi,t=100)
    # time evolution until acquisition
    M = np.matmul(rotMatrix_acq,M)+BB_acq
    # acquisition
    for j in range(acquisition_time//DW):
        M = np.matmul(rotMatrix_DW,M)+BB_DW
        XY[time] = np.sum(M[:,0,0])**2+np.sum(M[:,1,0])**2
        time = time + 1;
    # time evolution until next pi pulse
    M = np.matmul(rotMatrix_acq,M)+BB_acq

#plot results
plt.figure()
plt.plot(XY[0:time])

print(tim.perf_counter() - startTime)
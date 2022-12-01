import numpy as np
from scipy.io import loadmat

def rotZ(a):
    #returns rotation matrix for a left handed rotation by 
    #'a' radians about the z-axis
    return np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]])

def rotY(a):
    #returns rotation matrix for a left handed rotation by 
    #'a' radians about the y-axis
    return np.array([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]])

def rotX(a):
    #returns rotation matrix for a left handed rotation by 
    #'a' radians about the x-axis
    return np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]])

def lorentzian(x,a=100,c=1000):
    #create a lorenzian distribution with domain start to stop with num points
    y = a/((x/c)**2+1)
    return y

def sinc(x):
    y = np.zeros(len(x))
    y[x==0]=1
    y[x!=0]=np.sin(x[x!=0])/x[x!=0]
    return y

def rotAlpha(alpha, theta=0, phi=0):
    #returns rotation matrix for a rotation by alpha radians about an axis
    #defined by theta and phi
   
   return rotZ(-phi).dot(rotY(theta).dot(rotX(alpha).dot(rotY(-theta).dot(rotZ(phi)))))

def rotXY(alpha,phi=0):
    #performs a rotation by alpha degrees about an axis theta (+x to +y)
    cA = np.cos(alpha)
    sA = np.sin(alpha)
    cP = np.cos(phi)
    sP = np.sin(phi)
    
    return np.array([[sP**2*cA+cP**2,-sP*cA*cP+sP*cP,-sA*sP],
                     [-sP*cA*cP+sP*cP,sP**2+cA*cP**2,sA*cP],
                     [sA*sP,-sA*cP,cA]])

def relaxation(t,T1=100000,T2=10000):
    
    decay1=np.exp(-t/T1);
    decay2=np.exp(-t/T2);
    
    AA = np.array([[decay2,0,0],
                   [0,decay2,0],
                   [0,0,decay1]])
    BB = 1-decay1
    return (AA,BB)

def time_evolution_matrix(offset, time, T1, T2, M, isochromats=11):
    (AA,BB) = relaxation(time,T1,T2);
    mat = np.zeros((isochromats,3,3))
    for i in range(isochromats):
        mat[i,:,:] = rotZ(time * offset[i]*2*np.pi)
    mat = np.matmul(mat,AA)
    BB = M*BB
    
    return mat, BB

def rectPulse(alpha, M, rotMatrix, BB, phi=0, t=1):
    
    t = int(t)
    #create new rotation matrix then rotate for t timesteps
    newRotMatrix = np.matmul(rotMatrix,rotXY(alpha/t,phi=phi))
    xy=np.zeros(t)
    for i in range(t):
        M = np.matmul(newRotMatrix,M)+BB
        xy[i] = np.sum(M[:,0,0])**2+np.sum(M[:,1,0])**2
    return (M,xy)

def sincPulse(alpha, M, rotMatrix, BB, phi=0, t=50):
    k = 8*np.pi/t
    time = np.linspace(-t/2,t/2,t)
    a = alpha/sum(sinc(k*time))
    parts = a*sinc(k*time)
    xy=np.zeros(t)
    
    for i in range(t):
        newRotMatrix= np.matmul(rotMatrix,rotXY(parts[i],phi=phi))
        M = np.matmul(newRotMatrix,M)+BB
        xy[i] = np.sum(M[:,0,0])**2+np.sum(M[:,1,0])**2
    return (M,xy)

def slr5_180(alpha, M, rotMatrix, BB, phi=0, t=50):
    envelope = loadmat(r"C:\Users\marka\Documents\python_scripts\MRI\slr5_180.mat")['y'][0,:]
    from skimage.transform import resize
    envelope = resize(envelope, [t])
    
    time = np.arange(t)
    a = alpha/sum(envelope[time])
    
    parts = a*envelope[time]
    xy=np.zeros(t)
    
    for i in range(t):
        newRotMatrix= np.matmul(rotMatrix,rotXY(parts[i],phi=phi))
        M = np.matmul(newRotMatrix,M)+BB
        xy[i] = np.sum(M[:,0,0])**2+np.sum(M[:,1,0])**2
    return (M,xy)

def n_pulse(times, phi, alpha, T1=82e3, T2=71e3, rf_dur=5):
    isochromats = 201
    
    #create initial magnetization vectors with lorentzian distribution
    offset = np.linspace(-5000,5000,isochromats)
    M=np.zeros([isochromats,3,1])
    M[:,2,0]=lorentzian(offset,a=1,c=1000)
    
    #create time evolution matrix for 1us (precession from offset, T1/T2 relaxation)
    (AA,BB)=relaxation(1,T1,T2);
    rotMatrix = np.zeros((isochromats,3,3))
    for i in range(isochromats):
        rotMatrix[i,:,:]=rotZ(offset[i]*2*np.pi/10**6)
    rotMatrix = np.matmul(rotMatrix,AA)
    BB = M*BB
    
    #prealocate the xy-magnetization vector for 100000 timesteps
    XY = np.zeros(np.sum(times) + rf_dur*len(times))
    phase = np.zeros(np.sum(times) + rf_dur*len(times))
    time = 0
    
    for i in range(len(times)):
        # pulse
        (M,XY[time:time+rf_dur]) = rectPulse(alpha[i], M, rotMatrix, BB,
                                             phi=phi[i],t=rf_dur)
        time += rf_dur
        # time after pulse
        for j in range(times[i]):
            M = np.matmul(rotMatrix,M)+BB
            XY[time] = np.sum(M[:,0,0])**2+np.sum(M[:,1,0])**2
            phase[time] = np.angle( np.sum(M[:,0,0] + 1j*M[:,1,0]) )
            time = time + 1;
    
    return XY, phase

class PulseOptimiserSimulations():
    
    def __init__(self, isochromats=10, T1=500, T2=200):
        # maybe calculate distribution based on T2* in future
        self.offset = np.linspace(-500,500,isochromats)*2*np.pi
        self.M0=np.zeros([isochromats,3,1])
        self.M0[:,2,0]=lorentzian(self.offset,a=1,c=1000)
        self.offset = self.offset/10**6
        
        # started with inversion pulse
        self.M = -self.M0
        
        self.t1=T1
        self.t2=T2
        self.isochromats=isochromats
    
    def reset(self):
        self.M=self.M0
    
    def relaxation_time_step(self,t):
        #rotation
        rotMatrix = np.zeros((self.isochromats,3,3))
        for i in range(self.isochromats):
            rotMatrix[i,:,:]=rotZ(self.offset[i]*t)
        #relaxation
        (AA,BB)=relaxation(t,self.t1,self.t2);
        rotMatrix = np.matmul(rotMatrix,AA)
        BB = self.M0*BB
        
        self.M = np.matmul(rotMatrix,self.M)+BB
        
    def pulse(self,alpha):
        # assume hard pulse in x
        self.M = np.matmul(rotX(alpha),self.M)
        
    def get_signal(self):
        return np.sqrt(np.sum(self.M[:,0])**2 + np.sum(self.M[:,1])**2)

class mri_sim():
    
    def __init__(self, isochromats=10, T1=500000, T2=200000, T2_star=20000):
        # convert T2* to seconds
        self.offset = np.linspace(-3000,3000,isochromats)*2*np.pi #Hz
        self.M0=np.zeros([isochromats,3,1])
        self.M0[:,2,0]=lorentzian(self.offset,a=2*T2,c=1/(T2_star)**2)
        self.offset = self.offset/10**6 #to be compatible with us
        
        # initial magnetization
        self.M = self.M0
        
        self.t1=T1
        self.t2=T2
        self.isochromats=isochromats
    
    def reset(self):
        self.M=self.M0
    
    def relaxation_time_step(self,t):
        #rotation
        rotMatrix = np.zeros((self.isochromats,3,3))
        for i in range(self.isochromats):
            rotMatrix[i,:,:]=rotZ(self.offset[i]*t)
        #relaxation
        (AA,BB)=relaxation(t,self.t1,self.t2);
        rotMatrix = np.matmul(rotMatrix,AA)
        BB = self.M0*BB
        
        self.M = np.matmul(rotMatrix,self.M)+BB
        
    def pulse(self,alpha):
        # assume hard pulse in x
        self.M = np.matmul(rotX(alpha),self.M)
    
    def soft_pulse(self, alpha, phi, tau, dt):
        
        rotMatrix = np.zeros((self.isochromats,3,3))
        for i in range(self.isochromats):
            rotMatrix[i,:,:]=rotZ(self.offset[i])
        #relaxation
        (AA,BB)=relaxation(dt,self.t1,self.t2);
        rotMatrix = np.matmul(rotMatrix,AA)
        BB = self.M0*BB
        
        # alpha is flip angle in radiants, tau is the pulse duration in us
        (self.M, _) = rectPulse(alpha, self.M, rotMatrix, BB, phi=phi, t=tau/dt)
        
    def get_signal(self):
        return np.sqrt(np.sum(self.M[:,0])**2 + np.sum(self.M[:,1])**2)
    
    def spoiling(self):
        self.M[:,0:2,0] = 0
import time
import numpy as np
import matplotlib.pyplot as plt
import Sampler1
from Sampler1 import sqrtm
from Sampler1 import hmc, CHAINS
import matplotlib.pyplot as plt

rho = 1-1/10**14

#[[1   rho]
# [rho   1]]
def U(q):
    return (q[0]**2+q[1]**2 - 2*q[0]*q[1]*rho)/(1-rho**2)/2

def dU(q):
    return np.array([(2*q[0]-2*q[1]*rho)/(1-rho**2)/2,(2*q[1]-2*q[0]*rho)/(1-rho**2)/2])
def ddU(q):
    return np.array([[1/(1-rho**2),-rho/(1-rho**2)],[-rho/(1-rho**2),1/(1-rho**2)]])
if __name__ == '__main__':
    Sampler1.INTERVAL = 1001
    Sampler1.dt10 = 0.00000001
    Sampler1.dt20 = 0.00000001
    #np.random.seed(0)
    BURNIN = 2000
    EPISODE = 2000

    Dim = 2
    gen = hmc(U, dU,ddU, Dim, BURNIN, True, True)
    qs = []
    for j in range(EPISODE):
        for k in range(CHAINS):
            q = next(gen)
            qs.append(q)

    qs = np.array(qs)
    qs1 = np.transpose(np.linalg.solve(sqrtm(np.array([[1,rho],[rho,1]])),np.transpose(qs)))
    plt.figure()
    plt.plot(qs1[:,0],qs1[:,1], '.',alpha=0.5)
    plt.show()
    plt.pause(1000)

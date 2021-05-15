from math import sqrt, exp, log
from numpy.random import randn, rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#from autograd import grad
#import autograd.numpy as np
import numpy as np
#from autograd import hessian
import Sampler1
from Sampler1 import hmc, CHAINS,INTERVAL,sqrtm

if __name__ == '__main__':
    Sampler1.INTERVAL = 101
    np.random.seed(0)
    BURNIN = 500
    EPISODE = 1000
    Dim = 2
    rho = 1-1/10**9
    SIGMA = np.ones((Dim,Dim))
    SIGMA[0][0]= rho
    SIGMA[1][1] = rho

    U = lambda q: np.dot(q,np.linalg.solve(SIGMA,q))/2
    #dU = grad(U)
    dU = lambda q:np.linalg.solve(SIGMA,q)
    #ddU = hessian(U)
    ddU = lambda q:np.linalg.inv(SIGMA)
    gen = hmc(U, dU,ddU, Dim, BURNIN, True, True)
    qs = []
    for j in range(EPISODE):
        for k in range(CHAINS):
            q = next(gen)
            qs.append(q)
    qs = np.array(qs)
    qs1 = np.linalg.solve(sqrtm(SIGMA),qs.transpose())
    plt.figure()
    plt.plot(qs[:,0],qs[:,1], '.',alpha=0.5)
    plt.figure()
    plt.plot(qs1[0,:],qs1[1,:], '.',alpha=0.5)
    plt.show()
    plt.pause(1000)
 

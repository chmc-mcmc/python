import time
import numpy as np0
import matplotlib.pyplot as plt
from jax import grad, jit, jacfwd,jacrev
import jax.numpy as np
import Sampler1
from Sampler1 import hmc, CHAINS,sqrtm
import matplotlib.pyplot as plt

rho = 1-1/10**7
SIGMA = np0.array([[1,rho],[rho,1]])

@jit
def U(q):
    return np.dot(q,np.linalg.solve(SIGMA,q))/2

def hessian(f):
    return jit(jacfwd(jacrev(f)))

dU = grad(U)
ddU = hessian(U)

if __name__ == '__main__':
    Sampler1.INTERVAL = 101
    Sampler1.dt10 = 0.0001
    Sampler1.dt20 = 0.0001
    #np.random.seed(0)
    BURNIN = 500
    EPISODE = 1000

    Dim = 2


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

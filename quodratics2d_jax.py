import time
import numpy as np0
import matplotlib.pyplot as plt
from jax import grad, jit, jacfwd,jacrev
import jax.numpy as np
import Sampler1
from Sampler1 import hmc, CHAINS
import matplotlib.pyplot as plt

@jit
def U(q):
    return np.dot(q,np.linalg.solve(SIGMA,q))/2

def hessian(f):
    return jit(jacfwd(jacrev(f)))

dU = grad(U)
ddU = hessian(U)

if __name__ == '__main__':
    Sampler1.INTERVAL = 101
    Sampler1.dt10 = 0.00000001
    Sampler1.dt20 = 0.00000001
    #np.random.seed(0)
    BURNIN = 5000
    EPISODE = 10000
    rho = 1-1/10**6
    Dim = 2
    SIGMA = np0.ones((Dim,Dim))*rho
    np0.fill_diagonal(SIGMA,1)
    gen = hmc(U, dU,ddU, Dim, BURNIN, True, True)
    qs = []
    for j in range(EPISODE):
        for k in range(CHAINS):
            q = next(gen)
            qs.append(q)

    qs = np.array(qs)
    plt.figure()
    plt.plot(qs[:,0],qs[:,1], '.',alpha=0.5)
    plt.show()
    plt.pause(1000)

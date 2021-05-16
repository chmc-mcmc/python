import time
import Sampler1
from Sampler1 import hmc, CHAINS,INTERVAL
import gc

import torch
from hessian import hessian

def U(q0):
    if not torch.is_tensor(q0):
        q = torch.from_numpy(q0).to(torch.double)
    else:
        q = q0
    q.requires_grad=True
    q1 = q[:,None]
    q2,LU = torch.solve(q1,SIGMA)
    u= torch.dot(q1.flatten(),q2.flatten())/2
    return u

U1 = lambda q: U(q).detach().numpy()

def dU(q0):
    q = torch.from_numpy(q0).to(torch.double)
    q.requires_grad=True
    if q.grad != None:
        q.grad.data.zero_()
    u = U(q)
    u.backward()
    g = q.grad.numpy()
    return g

def ddU(q0):
    q = torch.from_numpy(q0).to(torch.double)
    q.requires_grad=True
    return hessian(U(q), q, create_graph=True).numpy()


if __name__ == '__main__':
    Sampler1.INTERVAL = 1001
    Sampler1.dt10 = 0.001
    Sampler1.dt20 = 0.001
    BURNIN = 15
    EPISODE = 15
    TPS = []
    for Dim in range(10,500,30):
        rho = 1-1/10**4
        SIGMA = torch.ones(Dim,Dim).to(torch.double)*rho
        torch.diagonal(SIGMA).fill_(1)
        t = time.time()
        gen = hmc(U1, dU,ddU, Dim, BURNIN, True, True)
        for j in range(EPISODE):
            for k in range(CHAINS):
                q = next(gen)
        elapsed = time.time() - t
        tps = elapsed/(CHAINS*(EPISODE+BURNIN))
        print("{} {}".format(Dim,tps))
        TPS.append(tps)
    gc.collect()
    print(TPS)

from numpy.random import randn, rand
import numpy as np

CHAINS = 5
STEPS = 5
dt10=0.0000001
dt20=0.0000001
INTERVAL=1001
outbnd=lambda q:False
decaydt = 0.1
decayenergy = 0.1
LOWAP = 0.1
HIGHAP = 0.9
switch0 = 100
diag = False

def sqrtm(x):
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    L = np.matmul(np.matmul(vh,np.diag(np.sqrt(np.abs(s)))),vh.T)
    return L

def hmc(U, dU, ddU, Dim, BURNIN, vanilla0, switch, qinit=None):
    dt1 = dt10
    dt2 = dt20
    vanilla = vanilla0
    if qinit==None:
        qAll = np.random.randn(CHAINS,Dim)

    j = 0
    Utotal = sum([U(qAll[i,:]) for i in range(CHAINS)])

    Htotal1 = 2*Utotal if Utotal>0 else 100
    Htotal2 = Htotal1
    while True:
        pAll = np.random.randn(CHAINS,Dim)
        KtotalNew = sum([np.dot(pAll[i,:],pAll[i,:])/2 if vanilla else np.dot(pAll[i,:],np.linalg.solve(ddU(qAll[i,:]), pAll[i,:]))/2   for i in range(CHAINS)])
        Utotal = sum([U(qAll[i,:]) for i in range(CHAINS)])
        if vanilla:
            Htotal = Htotal1
            dt = dt1
        else:
            Htotal = Htotal2
            dt = dt2
        Ktotal = Htotal - Utotal
        pAll = pAll * np.sqrt(abs(Ktotal/KtotalNew))
        ES = []
        AS = []

        for i in range(CHAINS):
            bad = False
            p = pAll[i,:]
            q = qAll[i,:]
            q0 = q
            if j < BURNIN:
                UE = [U(q)]
            for s in range(STEPS):
                p = p - dt * dU(q)
                q1 = q
                if vanilla:
                    q = q + dt * p
                else:
                    q = q + dt * np.linalg.solve(ddU(q), p)
                if outbnd(q):
                    q = q1
                    bad = True
                if j < BURNIN:
                    UE.append(U(q))
            if j< BURNIN:
                ES.append(UE)

            # metropolis
            if bad:
                alpha = 0
            else:
                alpha=np.exp(np.clip(U(q0)- U(q),-20,0))
            AS.append(alpha)
            if alpha < np.random.rand():
                q = q0
            qAll[i,:] = q
            if j >=BURNIN:
                yield q

        if j>=BURNIN:
            if j % INTERVAL==0:
                print(j,Ktotal,KtotalNew,Utotal, Htotal1,Htotal2,dt1,dt2, vanilla)
        else:
            ES = np.array(ES)
            AS = np.array(AS)

            S = np.unique(ES.argmax(axis=1))
            s = np.unique(ES.argmin(axis=1))

            if j % INTERVAL==0:
                print(j,Ktotal,KtotalNew,AS.mean(), AS.std(), Utotal, Htotal1,Htotal2,dt1,dt2, vanilla)

            if S.size==2 and s.size==2 and np.all(S==[0,STEPS]) and np.all(s==[0,STEPS]):
                dt = dt * (1+decaydt)
            elif np.all(s==[0]) and np.all(S==[STEPS]):
                dt = dt / (1+decaydt)
            hi = AS.mean() > HIGHAP
            lo = AS.mean() < LOWAP
            if lo:
                Htotal = (Htotal-Utotal)/(1+decayenergy)+Utotal
            elif hi:
                Htotal = (Htotal-Utotal)*(1+decayenergy)+Utotal
            if vanilla:
                dt1 = dt
                Htotal1 = Htotal
            else:
                dt2 = dt
                Htotal2 = Htotal
        if switch:
            vanilla = not vanilla
        j += 1

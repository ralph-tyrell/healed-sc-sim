import numpy as np
import matplotlib.pyplot as plt
from function_keeper import diosi_force, aV_maker, adV_maker, test
m = 1
pauli_z = np.array([[1, 0], 
                    [0, -1]])
lmbda = 1
sigma = 0.8#0.8
ltimes = 45000
phi = 2
gamma =  0.5
cov = [[gamma/2 , 0], [0, (1 /(2 * gamma))]]  #these are my mean and covariance matrices.  they're subject to change for ehrenfest dynamics
nsamples = 1 #this is the number of sample nuclei that will undergo the dynamics
n_snapshots = 1
fs_to_au =  41.341374575751 #conversion of fs to atomic units
sample_oc = np.zeros((nsamples, 2, n_snapshots+1))
a = np.zeros((n_snapshots, 20))
b = np.zeros((n_snapshots, 21))
dt = 1e-5
alpha = 1 
c = np.zeros((2,1),dtype=complex)

def coef_calculator(H, t, psiG, x, a, q):
 lmbda = 1
 sigma = 0.8
 eigenvalues, eigenvectors = np.linalg.eigh(H)
 dH = adV_maker(q)
 H = aV_maker(q)
 b = eigenvalues
 phi = np.array(np.copy(eigenvectors))
 phit = np.conjugate(phi.T)
 enmat = np.diag(np.exp(-b * t *1.0j))
 psiG= phi @ enmat @ phit.dot(psiG) -alpha *( ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) + ((lmbda**2/(8*sigma**2)) * ((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2)))@psiG))* t))))
 return psiG/ ((np.conjugate(psiG.T).dot(psiG))**0.5)  #these steps are still important due to their usefulness in the verlet_step
def verlet_step(q,p,m,dt,c):
  dW = test(dt)
  f = diosi_force(q, c) 
  H = aV_maker(q)
  c = coef_calculator(H, dt, c, dW, -f, q)
  q = q + (p/m) *dt
  p = p + f*dt + (alpha* sigma * dW)
  return q, p, c

p = np.zeros((ltimes+1, 1))
q = np.zeros((ltimes+1, 1))
for i in range(nsamples):
  c = np.zeros((2,1),dtype=complex)
  c[0,0] = 1/(2**0.5)
  c[1,0] = 1/(2**0.5)
  p[0] = 10 
  q[0] = 0 
  for l in range(ltimes):
    q[l+1], p[l+1], c = verlet_step(q[l], p[l], m, dt, c) #makes a list of ltimes snapshots)
    print(c*np.conjugate(c)) #multiple trials of this should yield the p 
x = np.c_[q,p]
plt.plot(q, p)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from function_keeper import diosi_force, aV_maker, adV_maker, test#, #force#, coef_calculator# delta, dkappa, d_delta, V_maker, dV_maker,test, force, coef_calculator, wavefunction_solver
#from tdhamiltonian import wavefunction_solver
#def sech(x):
#    return 1 / np.cosh(x)
m = 1
pauli_z = np.array([[1, 0], 
                    [0, -1]])
lmbda = 1
sigma = 0.8#0.8
ltimes = 45000
phi = 2
#p = np.zeros((ltimes+1, 1))
#q = np.zeros((ltimes+1, 1))
#mean = [(120**0.5),  -15]  # Mean vector (mu_x, mu_y)
gamma =  0.5
cov = [[gamma/2 , 0], [0, (1 /(2 * gamma))]]  #these are my mean and covariance matrices.  they're subject to change for ehrenfest dynamics
nsamples = 1 #this is the number of sample nuclei that will undergo the dynamics
n_snapshots = 1
fs_to_au =  41.341374575751 #conversion of fs to atomic units
sample_oc = np.zeros((nsamples, 2, n_snapshots+1))
a = np.zeros((n_snapshots, 20))
b = np.zeros((n_snapshots, 21))
dt = 1e-5# * 
alpha = 1 
c = np.zeros((2,1),dtype=complex)
#c[0,0] = 1/(2**0.5)
#c[1,0] = 1/(2**0.5)
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
# psiG= phi @ enmat @ phit.dot(psiG) - alpha*( ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) + ((lmbda**2/(8*sigma**2)) * ((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2)))@psiG))* t))))
 psiG= phi @ enmat @ phit.dot(psiG) -alpha *( ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) + ((lmbda**2/(8*sigma**2)) * ((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2)))@psiG))* t))))
 return psiG/ ((np.conjugate(psiG.T).dot(psiG))**0.5)#psiG  #return x**#2
def verlet_step(q,p,m,dt,c):
  dW = test(dt)
  f = diosi_force(q, c) 
  H = aV_maker(q)
  c = coef_calculator(H, dt, c, dW, -f, q)
  q = q + (p/m) *dt
  p = p + f*dt + (alpha* sigma * dW)
  return q, p, c
# the next section is added to accurately model a bunch of particles being ran through this simulation
p = np.zeros((ltimes+1, 1))
q = np.zeros((ltimes+1, 1))
for i in range(nsamples):
  c = np.zeros((2,1),dtype=complex)
  c[0,0] = 1/(2**0.5)
  c[1,0] = 1/(2**0.5)
  #random_sample = np.random.multivariate_normal(mean, cov)
  p[0] = 10 #random_sample[0] #random_sample[0]
  q[0] = 0 #random_sample[1]  #chooses i random samples of position and momentum
  for l in range(ltimes):
    q[l+1], p[l+1], c = verlet_step(q[l], p[l], m, dt, c) #makes a list of ltimes snapshots)
    print(c*np.conjugate(c))
x = np.c_[q,p]
#np.savetxt("up.txt", x, delimiter=',')

plt.plot(q, p)
plt.show()
#  for n in range(n_snapshots):
#     sample_oc[i, 0, n] = q[-1, 0]  #makes n_snapshots of the arrays.
#for n in range(n_snapshots):
#  a[n, :], b[n, :] = np.histogram(sample_oc[:,0, n], bins=20, range=None, density=True, weights=None) #makes nsnapshots of graphs
#for n in range(n_snapshots):
#  plt.plot(b[n, :-1], a[n,:]) #compiles and plots snapshots
#ca = np.loadtxt("ref_data.py", delimiter=",")
#print(ca[0,0])
#print(ca[0,1])
#plt.plot(ca[:,0], ca[:,1])
#j = np.loadtxt("exact_result.txt", delimiter=",")
#plt.plot(j[:,0], j[:,1])
#plt.show()

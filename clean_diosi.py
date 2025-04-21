import numpy as np
import matplotlib.pyplot as plt
from function_keeper import kappa, delta, dkappa, d_delta, V_maker, dV_maker,test, true

m = 2000
pauli_z = np.array([[1, 0], 
                    [0, -1]])
alpha = 1
ltimes = 1500
mean = [(120**0.5),  -15]  # Mean vector (mu_x, mu_y)
gamma =  0.5
cov = [[gamma/2 , 0], [0, (1 /(2 * gamma))]]  #these are my mean and covariance matrices.  they're subject to change for ehrenfest dynamics
nsamples = 1000 #this is the number of sample nuclei that will undergo the dynamics
n_snapshots = 1
fs_to_au =  41.341374575751 #conversion of fs to atomic units
sample_oc = np.zeros((nsamples, 2, n_snapshots+1))
a = np.zeros((n_snapshots, 20))
b = np.zeros((n_snapshots, 21))
dt = 0.1 * fs_to_au# * 
c = np.zeros((2,1),dtype=complex)
#c[0,0] = 1
#c[1,0] = 0
def coef_calculator(H, t, psiG, x, a, q):
 lmbda = 1
 sigma = true(q)
 eigenvalues, eigenvectors = np.linalg.eigh(H)
 dH = dV_maker(q)
 b = eigenvalues
 phi = np.array(np.copy(eigenvectors))
 phit = np.conjugate(phi.T)
 enmat = np.diag(np.exp(-b * t *1.0j))
 psiG= phi @ enmat @ phit.dot(psiG) -alpha *( ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) + ((lmbda**2/(8*sigma**2)) * (((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2))))@psiG))* t))))
 return psiG/ ((np.conjugate(psiG.T).dot(psiG))**0.5) #this was the finished product of my simulations.  This calculation ultimately mixed Ehrenfest dynamics with stochastic noise, we tweaked how "sigma" interacted with the dynamics due to the fact that our chemical system and the system proposed in the paper were different.

def force(q, c):
  dH = dV_maker(q)
  c_t =  np.conjugate(c.T)
  force = ((c_t @ dH).dot(c)) * -1/ (c_t.dot(c))
  return np.real(force) #this was the Langevin equation for force

def verlet_step(q,p,m,dt,c):
  dW = test(dt)
  sigma = true(q)
  f = force(q, c) 
  H = V_maker(q)
  c = coef_calculator(H, dt, c, dW, -f, q )
  q = q + (p/m) *dt
  p = p + f*dt + (alpha * sigma * dW)
  return q, p, c
# the next section is added to accurately model a bunch of particles being ran through this simulation
p = np.zeros((ltimes+1, 1))
q = np.zeros((ltimes+1, 1))
for i in range(nsamples):
  c = np.zeros((2,1),dtype=complex)
  c[0,0] = 1
  c[1,0] = 0
  random_sample = np.random.multivariate_normal(mean, cov)
  p[0] = random_sample[0] #random_sample[0]
  q[0] = random_sample[1]  #chooses i random samples of position and momentum
  for l in range(ltimes):
    q[l+1], p[l+1], c = verlet_step(q[l], p[l], m, dt, c) #makes a list of ltimes snapshots)
#print(p, q)
  for n in range(n_snapshots):
     sample_oc[i, 0, n] = q[-1, 0]  #makes n_snapshots of the arrays.
for n in range(n_snapshots):
  a[n, :], b[n, :] = np.histogram(sample_oc[:,0, n], bins=20, range=None, density=True, weights=None) #makes nsnapshots of graphs
print(a)
print(b)
np.reshape(a, 20)
np.reshape(b[:,:-1], 20)
x = np.c_[a[0,:],((b[0,:-1]+b[0,1:])/2)]
np.savetxt("sigma_0_05.txt", x)
for n in range(n_snapshots):
  plt.plot(((b[n,:-1]+b[n,1:])/2), a[n,:], label = 'Position-Dependent Sigma') #compiles and plots snapshots

ca = np.loadtxt("ref_data.py", delimiter=",")
print(ca[0,0])
print(ca[0,1])
plt.plot(ca[:,0], ca[:,1], label = 'Ehrenfest', color = 'purple')
j = np.loadtxt("exact_result.txt", delimiter=",")
plt.plot(j[:,0], j[:,1], label='Exact Result', linestyle='dashdot', color = "black")
plt.xlabel('q')  # Label for the x-axis
plt.ylabel('Probability')  # Label for the y-axis
plt.title('Comparison of Results') 
plt.legend()
plt.show()

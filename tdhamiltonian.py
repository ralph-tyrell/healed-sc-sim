import numpy as np
import matplotlib.pyplot as plt
dt=np.pi/100000
ntimes = 100000
t = np.zeros(ntimes+1)
H = np.zeros((2,2), dtype=complex)
delta = 1
mu = 1
epsilon = 0.1 #define the elements of TIH
H[0,0] = delta
H[1,0] = mu * epsilon
H[0,1] = mu * epsilon
H[1,1] = -delta
t = np.linspace(0, dt * ntimes, ntimes + 1)
psiA =  np.zeros((2, ntimes+1),dtype=complex) #defines the initial vector for the ground statei
psiB =  np.zeros((2, ntimes+1),dtype=complex) #defines the initial vector for the ground stat

psiA[0, 0] = 1
psiA[0, 1] = 0
psiB[0, 0] = 0
psiB[0, 1] = 1

analytical_solution = np.zeros(ntimes+1)
def timedependent_H(t):
  epsilon_t = 2 * epsilon * np.cos(2 * delta * t)
  H[0,0] = delta
  H[1,0] = mu * epsilon_t
  H[0,1] = mu * epsilon_t
  H[1,1] = -delta
  return H
def wavefunction_solver(H, t, psiA):  #this will solve the time propagated version of the ground state
  eigenvalues, eigenvectors = np.linalg.eigh(H)
  a = eigenvalues
  phi = np.array(np.copy(eigenvectors))
  phit = np.transpose(phi)
  enmat = np.diag(np.exp(-a * t *1.0j))
  psiA= phi @ enmat @ phit.dot(psiA)
  return psiA
def coef_calculator(H, t, psiG, x, a, lmbda, sigma):
  b = wavefunction_solver(H, t, psiG)
  return a + ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) - ((lmbda**2/(8*sigma**2)) * ((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2)))@psiG))* t)))
for i in range(ntimes):
  H = timedependent_H(t[i])
  psiA[:,i+1] = wavefunction_solver(H, dt, psiA[:,i])
a = np.abs((np.transpose(psiA))**2)
for i in range(ntimes):
  H = timedependent_H(t[i])
  psiB[:,i+1] = wavefunction_solver(H, dt, psiB[:,i])
d = np.abs((np.transpose(psiB))**2)
b = a[:, 0]
c = a[:, 1]
e = d[:, 0]
f = d[:, 1]
for i in range(ntimes + 1):
  analytical_solution[i] = (np.cos(mu * epsilon *t[i])**2)
plt.plot(t, analytical_solution, label='analytical solution')
g = e + b
h = f + c
plt.plot(t, g, label='Pe(t)')
plt.ylabel("Probability be in State")
plt.xlabel("Time Steps")
plt.title('eps = 0.1')
plt.legend(loc='upper right', fontsize='large', shadow=True, title='Legend')
plt.show()
plt.show()
plt.show()

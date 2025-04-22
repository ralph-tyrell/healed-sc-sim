import numpy as np
import matplotlib.pyplot as plt
ntimes = 30000
t = np.zeros(ntimes+1)
H = np.zeros((2,2))
delta = 0.0000493
E = 1
H[0,0] = E
H[1,0] = -delta
H[0,1] = -delta
H[1,1] = E
psiG =  np.zeros((2, ntimes+1),dtype=complex) #defines the initial vector for the ground state
psiE = np.zeros((2, ntimes+1), dtype=complex) #defines the initial vector for the excited state 
psiG[0,0] = -1/(2**0.5)
psiG[0,1] = -1/(2**0.5)
psiE[0,0] = -1/(2**0.5)
psiE[0,1] = 1/(2**0.5) #this defines the starting vector for both vectors
dt=np.pi*2
time = np.zeros(ntimes+1)
a, b =np.linalg.eigh(H)
print(b)
def wavefunction_solver(H, t, psiG):  #this will solve the time propagated version of the ground state.
  eigenvalues, eigenvectors = np.linalg.eigh(H)
  a = eigenvalues
  phi = np.array(np.copy(eigenvectors))
  phit = np.transpose(phi)
  enmat = np.diag(np.exp(-a * t *1.0j))
  psiG= phi @ enmat @ phit.dot(psiG)
  return psiG
#H[0,0] = E
#H[1,0] = delta
#H[0,1] = delta
#H[1,1] = E
def wavefunction_solver(H, t, psiE):
  H[0,0] = E
  H[1,0] = delta
  H[0,1] = delta
  H[1,1] = E  #inputs new hamiltonian, which has a + delta instead of a - delta
  eigenvalues, eigenvectors = np.linalg.eigh(H)
  a = eigenvalues
  phi = np.array(np.copy(eigenvectors))
  phit = np.transpose(phi)
  enmat = np.diag(np.exp(-a * t *1.0j))
  psiE= phi @ enmat @ phit.dot(psiE)
  return psiE
#def time_vector(dt):
#  x = 0
#  dt = np.pi/100
#  z= dt*a
#  x  += 1
#  return  z
for i in range(ntimes):
  psiG[:,i+1] = wavefunction_solver(H, dt, psiG[:,i])
#print(np.transpose(psiG)**2)
for i in range(ntimes):
  psiE[:,i+1] = wavefunction_solver(H, dt, psiE[:,i])
for i in range(ntimes):  #finish this, it should produce a loop
  time[i+1] = time[i] + dt
#print(time)  

#print(np.transpose(psi))
a = (np.abs((np.transpose(psiG)))**2)
#print(a)
c = a[:,1]
b = (np.abs(np.transpose(psiE))**2)
#print(c.shape)
#print(c)

real_psiG_up = np.real(c) #takes the up component of the graph
#plt.plot(time, real_psiG)
d = a[:,0]
real_psiG_down = np.real(d) 
p = b[:,0]
q = b[:,1]
real_psiE_down = np.real(p)
real_psiE_up = np.real(q)
w = real_psiE_up + real_psiG_down
print(w)
plt.plot(time, real_psiE_down + real_psiG_down)
#plt.plot(time, real_psiE)
#plt.show()
plt.plot(time, real_psiE_up + real_psiG_up)
plt.show()
#print(a+b)
#plt.plot([a+b])
plt.show()

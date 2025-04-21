import numpy as np
def true(x):
 a = (0.015*np.exp(-x**2))+0.0000001
 return a
def sech(x):
    return 1 / np.cosh(x)
def kappa(x):
 return np.real(0.01 * np.tanh(1.6*x))
def delta(x):
  return np.real(0.005 * np.exp(-x**2))
def dkappa(x):
 return np.real(0.016 * (sech(1.6 * x)**2))
def d_delta(x):
  return np.real(-0.005*2*x * np.exp(-x**2))
def V_maker(x):
  V =  np.zeros((2,2))
  V[0,0] = kappa(x)
  V[1,0] = delta(x)
  V[0,1] = delta(x)
  V[1,1] = -kappa(x)
  return V
def dV_maker(x):
#  dV = np.array([[dkappa(x), d_delta(x)], 
#                [d_delta(x), -dkappa(x)]])
  if isinstance(x, np.ndarray):
    dV = np.zeros((2,2))
    dV[0,0] = dkappa(x[0])
    dV[1,0] = d_delta(x[0])
  
    dV[0,1] = d_delta(x[0])
    dV[1,1] = -dkappa(x[0])
  else:
    dV = np.zeros((2,2))
    dV[0,0] = dkappa( x)
    dV[1,0] = d_delta(x)
 
    dV[0,1] = d_delta(x)
    dV[1,1] = -dkappa(x)
  return dV
def aV_maker(x):
  lmbda = 1
  phi =2
  aV =  np.zeros((2,2))
  aV[0,0] = 2* lmbda* x+ phi
  aV[1,0] = 0
  aV[0,1] = 0
  aV[1,1] = -2* lmbda* x- phi
  return aV
def adV_maker(x):
#  dV = np.array([[dkappa(x), d_delta(x)], 
#                [d_delta(x), -dkappa(x)]])
  lmbda = 1
  phi = 2
  if isinstance(x, np.ndarray):
    adV = np.zeros((2,2))
    adV[0,0] = 2 * lmbda
    adV[1,0] = 0

    adV[0,1] = 0
    adV[1,1] = -2 *lmbda
  else:
    adV = np.zeros((2,2))
    adV[0,0] = 2 * lmbda
    adV[1,0] = 0

    adV[0,1] = 0
    adV[1,1] = -2 * lmbda
  return adV

mean = 0   # 
 # 
size = 1  
def force(q, c):
  dH = dV_maker(q)
  c_t =  np.conjugate(c.T)
  force = (((c_t @ dH).dot(c)) * -1)/ (c_t.dot(c)) # this is where we're still working but it should be a dot product
  return np.real(force)
def diosi_force(q, c):
  dH = adV_maker(q)
  c_t =  np.conjugate(c.T)
  force = (((c_t @ dH).dot(c)) * -1)/ (c_t.dot(c)) # this is where we're still working but it should be a dot product
  return np.real(force)
def coef_calculator(H, t, psiG, x, a):
 lmbda = 1
 sigma = 0.0001
 eigenvalues, eigenvectors = np.linalg.eigh(H)
 dH = dV_maker(q)
 b = eigenvalues
 phi = np.array(np.copy(eigenvectors))
 phit = np.conjugate(phi.T)
 enmat = np.diag(np.exp(-b * t *1.0j))
 psiG= phi @ enmat @ phit.dot(psiG) + ((lmbda/2*sigma) * (((dH - ((a)* np.identity(2))) @ psiG) *x)) - ((lmbda**2/(8*sigma**2)) * ((((dH - ((a)* np.identity(2))) @ ((dH - ((a)* np.identity(2)))@psiG))* t)))
 return psiG#psiG  #return x**#2

def wavefunction_solver(H, t, psiA):  #this will solve the time propagated version of the ground state
  eigenvalues, eigenvectors = np.linalg.eigh(H)
  a = eigenvalues
  phi = np.array(np.copy(eigenvectors))
  phit = np.transpose(phi)
  enmat = np.diag(np.exp(-a * t *1.0j))
  psiA= phi @ enmat @ phit.dot(psiA)
  return psiA
def coef_calculator(H, t, psiG, x, a, q):
  lmbda = 1
  sigma = 0.0001
  dH = dV_maker(q)
  b = wavefunction_solver(H, t, psiG)
  if b is None or lmbda is None or sigma is None or dH is None or a is None or psiG is None or x is None or t is None:
        raise ValueError("One of the input variables is None. Please initialize all variables.")
  identity_matrix = np.identity(2)
  term1 = lmbda / (2 * sigma) * ((dH - a * identity_matrix) @ psiG) * x
  term2 = lmbda**2 / (8 * sigma**2) * ((dH - a * identity_matrix) @ ((dH - a * identity_matrix) @ psiG)) * t
  result = b + term1 - term2
  return result
# Generate random numbers from the normal distribution
def test(dt):
 std_dev = dt**0.5
 a  = np.random.normal(0, std_dev, 1)
 return a
print(dV_maker(2))

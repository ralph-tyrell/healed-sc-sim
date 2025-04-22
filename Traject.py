import numpy as np
import matplotlib.pyplot as plt
dt = 0.1
m = 1
ntimes = 100
p = np.zeros(ntimes+1)
x = np.zeros(ntimes+1)
x[0] = -5
p[0] = 1
def potential(x):
  return np.exp(-x**2)  #return x**2
def force(x):
  return 2*x*np.exp(-x**2)
def verlet_step(x,p,m,dt):
  f = force(x)
  p = p+0.5*f*dt
  x = x+(p/m)*dt
  f=force(x)
  p = p+0.5*f*dt
  return x, p
for i in range(ntimes):
  x[i+1], p[i+1] = verlet_step(x[i],p[i],m,dt)
plt.plot(np.arange(ntimes+1)*dt,x, color='red')
plt.plot (np.arange(ntimes+1)*dt, potential(x), color='blue')
plt.plot (np.arange(ntimes+1)*dt, p**2/2+potential(x), color='green')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import animation
from matplotlib import rc
from IPython.display import HTML

class VectorField:
  def __init__(self, parametric_curve, simulation_time, Ts, vr=.25, Kf=10):
    self.parametric_curve = parametric_curve
    self.simulation_time = simulation_time
    self.Ts = Ts
    
    self.vr = vr
    self.Kf = Kf

    self.granularity = 1000

    self.fig = plt.figure(figsize=(10, 10))
    self.ax = plt.axes(projection='3d')

  def compute(self, p, t):
    s = np.linspace(0, 2*np.pi, self.granularity)

    curve = self.parametric_curve(s,t)

    distances = np.linalg.norm(curve - p, axis=1)
    closest_point_index = np.argmin(distances)
    closest_point = curve[closest_point_index,:]

    # Convergence parameters
    D = distances[closest_point_index]
    D_vec = p - closest_point
    D_unit = D_vec/(D+1e-6)
    # print(closest_point_index)
    # print(D_unit)

    # Tangent parameters
    s_star = s[closest_point_index]
    T = (self.parametric_curve(s_star+1e-3,t) - self.parametric_curve(s_star-1e-3,t)) / (2e-3)
    T = T/np.linalg.norm(T)
    Pi = np.eye(3) - T.dot(T.T)
    # print(T)

    # Feedforward parameters    
    forward_curve = self.parametric_curve(s,t+self.Ts)

    forward_distances = np.linalg.norm(forward_curve - p, axis=1)
    forward_closest_point_index = np.argmin(forward_distances)
    forward_closest_point = forward_curve[forward_closest_point_index,:]

    forward_D_vec = p - forward_closest_point # CHANGED THIS ORDER
    delta_D = (forward_D_vec - D_vec)/self.Ts
    # print(delta_D)

    # Modulation parameters
    G = (2/np.pi)*np.arctan(self.Kf*D)
    H = np.sqrt(1-G*G)
    # print(G, D_unit)
    # print(H)

    Phi_T = -Pi.dot(delta_D)
    Phi_S = -G*D_unit + H*T
    
    # print(Pi, delta_D)
    # print((Phi_S.dot(Phi_T))**2, self.vr**2, np.linalg.norm(Phi_T)**2)

    eta = -Phi_S.dot(Phi_T) + np.sqrt((Phi_S.dot(Phi_T))**2 + self.vr**2 - np.linalg.norm(Phi_T)**2)

    # Compute field
    Phi = eta*Phi_S + Phi_T

    return Phi

  def frameField(self, n):
    self.ax.clear()

    self.ax.set_xlim3d(-1,1)
    self.ax.set_ylim3d(-1,1)
    self.ax.set_zlim3d(-1,1)

    x, y, z = np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
    t = n*self.Ts
    s = np.linspace(0, 2*np.pi, self.granularity)
    path = self.parametric_curve(s,t)

    self.ax.set_xlabel('X')
    self.ax.set_ylabel('Y')
    self.ax.set_zlabel('Z')

    for i in range(0,10):
      for j in range(0,10):
        for k in range(0,10):
          u, v, w = MyField.compute(p=np.array([x[i],y[j],z[k]]), t=t)
          self.ax.quiver(x[i], y[j], z[k], u, v, w, length=0.1)

    self.ax.plot(path[0:,0], path[0:,1], path[0:,2])
    return self.ax

  def animateField(self):
    rc('animation', html='jshtml')
    
    anim = animation.FuncAnimation(self.fig, self.frameField, frames=int(self.simulation_time/self.Ts), blit=False, repeat=True, interval=1000*self.Ts)

    return HTML(anim.to_html5_video())

  def frameReference(self, n):
    self.ax.clear()

    self.ax.set_xlim3d(-1,1)
    self.ax.set_ylim3d(-1,1)
    self.ax.set_zlim3d(-1,1)

    s = np.linspace(0,2*np.pi, self.granularity)
    t = n*self.Ts
    path = self.parametric_curve(s,t)

    self.ax.plot(path[0:,0], path[0:,1], path[0:,2], label='parametric curve')

    return self.ax

  def animateReference(self):
    rc('animation', html='jshtml')
    
    anim = animation.FuncAnimation(self.fig, self.frameReference, frames=int(self.simulation_time/self.Ts), blit=False, repeat=True, interval=1000*self.Ts)

    return HTML(anim.to_html5_video())
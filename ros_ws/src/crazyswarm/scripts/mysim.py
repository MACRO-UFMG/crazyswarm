#!/usr/bin/env python3
import time
import numpy as np
import os
import sys
import rospy

# %% Setup Paths
# Keep track of the file path
file_path = os.path.dirname(os.path.realpath(__file__))
# Get project /src folder
src_path = os.path.dirname(os.path.dirname(file_path))
# Point to /crazyswarm/scripts
cs_scripts_path = os.path.join(src_path, "crazyswarm/scripts")
# %% Import Crazyflie
# Change working dir to /crazyswarm/scripts
os.chdir(cs_scripts_path)
sys.path.append(cs_scripts_path)
# Then we can import Crazyswarm lib
from pycrazyswarm import Crazyswarm


TAKEOFF_HEIGHT = 0.6
TAKEOFF_DURATION = 2.5
HOVER_DURATION = 1.0

swarm = Crazyswarm()
timeHelper = swarm.timeHelper
cf = swarm.allcfs.crazyflies[0]

FLY_DRONE=False

class CurveVectorField:
  
  def __init__(self, parametric_curve, simulation_time, Ts=.1, vr=.25, kG=10):
    self.parametric_curve = parametric_curve
    self.simulation_time = simulation_time
    self.Ts = Ts
    
    self.vr = vr
    self.kG = kG

    self.granularity = 1000

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
    G = (2/np.pi)*np.arctan(self.kG*D)
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


class ObstacleVectorField:
    
    def __init__():
        pass

    def compute():
        pass


class VectorField:

    def __init__(self, parametric_curve, simulation_time, Ts=.1, vr=.25, kG=10):
        self.curve_vector_field = CurveVectorField(parametric_curve, simulation_time, Ts, vr, kG)
        self.obstacle_vector_field = ObstacleVectorField()
        self.Ts = Ts


def takeoff():
    if FLY_DRONE:
        cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)

def control():
    curve = lambda s,t: np.array([0.6*np.cos(s), 0.6*np.sin(s), 1.0+0*s+0.2*np.cos(0.2*t)]).T
    cvf = CurveVectorField(parametric_curve=curve, simulation_time=60)
    while not rospy.is_shutdown():
        p = cf.position()
        t = timeHelper.time()
        v = cvf.compute(p, t)
        if FLY_DRONE:
            cf.cmdVelocityWorld(v, yawRate=0)
        else:
            print(v)
        time.sleep(0.1)

def land():
    if FLY_DRONE:
        cf.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(TAKEOFF_DURATION)
        cf.notifySetpointsStop()


def main():
    rospy.init_node('my_controller')
    print("Taking off...")
    takeoff()
    try:
        print("Controlling...")
        control()
    finally:
        print("Landing...")
        land()


if __name__ == "__main__":
    main()

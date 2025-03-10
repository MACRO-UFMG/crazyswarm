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
  
    def __init__(self, parametric_curve, Ts=.1, vr=.25, kG=10):
        self.parametric_curve = parametric_curve
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
        s_star = s[closest_point_index]

        # Convergence parameters
        D = distances[closest_point_index]
        D_vec = p - closest_point
        D_unit = D_vec/(D+1e-6)

        # Tangent parameters
        T = (self.parametric_curve(s_star+1e-3,t) - self.parametric_curve(s_star-1e-3,t)) / (2e-3)
        T = T/np.linalg.norm(T)
        Pi = np.eye(3) - T.dot(T.T)

        # Feedforward parameters    
        forward_curve = self.parametric_curve(s,t+self.Ts)

        forward_distances = np.linalg.norm(forward_curve - p, axis=1)
        forward_closest_point_index = np.argmin(forward_distances)
        forward_closest_point = forward_curve[forward_closest_point_index,:]
        forward_D_vec = p - forward_closest_point
        delta_D = (forward_D_vec - D_vec)/self.Ts

        # Modulation parameters
        G = (2/np.pi)*np.arctan(self.kG*D)
        H = np.sqrt(1-G*G)

        # Compute field
        Phi_S = -G*D_unit + H*T
        Phi_T = -Pi.dot(delta_D)
        eta = -Phi_S.dot(Phi_T) + np.sqrt((Phi_S.dot(Phi_T))**2 + self.vr**2 - np.linalg.norm(Phi_T)**2)
        Phi = eta*Phi_S + Phi_T
        return Phi


class ObstacleVectorField:
    
    def __init__(self, lam=0.1, M = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), vr=.25, kG=10):
        self.lam = lam # lambda parameter
        self.vr = vr
        self.kG = kG
        self.M = M

    def compute(self, p):
        # Smooth distance
        # Do = compute_gradient( @(x) smooth_distance(x,h,O), p)
        Do = 100*np.array([1,0,0])
        o_star = p - Do
        # persistent o_star_prev;
        # if isempty(o_star_prev)
        #     o_star_prev = o_star;
        # end
        # do_stardt = (o_star - o_star_prev)/dt;
        # o_star_prev = o_star;
        do_stardt = 0*np.array([1,0,0])

        # Field components
        Dlam = Do - self.lam*Do/np.linalg.norm(Do)
        Tlam = self.M.dot(Do)

        # Modulation parameters
        G = (2/np.pi)*np.arctan(self.kG*Dlam)
        H = np.sqrt(1-G*G)

        # Compute field
        Psi_S = -G*Dlam/np.linalg.norm(Dlam+1e-6) + H*Tlam/np.linalg.norm(Tlam+1e-6)
        Psi_T = do_stardt
        eta = -Psi_S.dot(Psi_T) + np.sqrt((Psi_S.dot(Psi_T))**2 + self.vr**2 - np.linalg.norm(Psi_T)**2)
        Psi = eta*Psi_S + Psi_T

        self.Do = Do
        self.Psi_T = Psi_T
        return Psi

    def get_distance_vector(self):
        return self.Do
    
    def get_feedforward_vector(self):
        return self.Psi_T


class VectorField:

    def __init__(self,
                 parametric_curve, Ts=.1, vr=.25, kG=10, # curve field parameters
                 lam=0.1, M = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), vr_o=.25, kG_o=10, # obstacle field parameters
                 B_crit=0.01, B_safe=0.1, kB=10 # blending parameters
                 ):
        self.curve_vector_field = CurveVectorField(parametric_curve, Ts, vr, kG)
        self.obstacle_vector_field = ObstacleVectorField(lam, M, vr_o, kG_o)
        self.Ts = Ts
        self.lam = lam # lambda parameter
        self.B_crit = B_crit
        self.B_safe = B_safe
        self.kB = kB

    def compute(self, p, t):
        # Compute fields
        curve = self.curve_vector_field.compute(p, t)
        obstacle = self.obstacle_vector_field.compute(p)
        Do = self.obstacle_vector_field.get_distance_vector()
        do_stardt = self.obstacle_vector_field.get_feedforward_vector()
        # Barrier functions
        B = 0.5*np.linalg.norm(Do)*np.linalg.norm(Do) + 0.5*self.lam*self.lam
        dot_B = Do.dot(curve - do_stardt)
        # Blend fields
        Theta1 = min(1,max(0,
                           (B-self.B_crit)/(self.B_safe - self.B_crit)
                           ))
        Theta2 = min(1,max(0,
                           self.kB*dot_B
                           ))
        Theta = min(1,Theta1+Theta2)
        F = Theta*curve + (1-Theta)*obstacle
        return F


def takeoff():
    if FLY_DRONE:
        cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)

def control():
    curve = lambda s,t: np.array([0.6*np.cos(s), 0.6*np.sin(s), 1.0+0*s+0.2*np.cos(0.2*t)]).T
    vf = VectorField(curve)
    while not rospy.is_shutdown():
        p = cf.position()
        t = timeHelper.time()
        v = vf.compute(p, t)
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

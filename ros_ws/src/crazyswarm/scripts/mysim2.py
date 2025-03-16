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


def compute_gradient(f, x):
    x = x.flatten()
    epsilon = 1e-6
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        dp = np.zeros(n)
        dp[i] = epsilon
        f_plus = f(x + dp)
        f_minus = f(x - dp)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad


def smooth_distance(p, h, A):
    m_samples = A.shape[1]
    if not m_samples:
        return np.inf   
    D = 0
    for i in range(m_samples):
        a = A[:,i].T
        D = D + (np.linalg.norm(p-a)**2/2)**(-1/h)
    D = D**(-h)
    return D


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
        Pi = np.eye(6) - T.dot(T.T)

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
    
    def __init__(self, lam=0.2, M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), h = 0.15, vr=.25, kG=10):
        self.lam = lam # lambda parameter
        self.M = M
        self.h = h
        self.vr = vr
        self.kG = kG
        self.__prev_t = None
        self.__prev_o_star = None
        self.O = np.matrix([])

    def compute(self, p, t):
        # Smooth distance
        Do = compute_gradient(lambda x: smooth_distance(x, self.h, self.O), p)
        # Do = 100*np.array([1,0,0])

        # Compute do_star/dt
        o_star = p - Do
        if self.__prev_t is None:
            self.__prev_t = t - 0.1
        dt = t - self.__prev_t
        self.__prev_t = t

        if self.__prev_o_star is None:
            self.__prev_o_star = o_star
        do_stardt = (o_star - self.__prev_o_star)/dt
        self.__prev_o_star = o_star
        # do_stardt = 0*np.array([1,0,0])

        # Field components
        Dlam = Do - self.lam*Do/np.linalg.norm(Do)
        Tlam = -self.M.dot(Do)

        # Modulation parameters
        G = (2/np.pi)*np.arctan(self.kG*np.linalg.norm(Dlam))
        H = np.sqrt(1-G*G)

        # Compute field
        Psi_S = -G*Dlam/np.linalg.norm(Dlam+1e-6) + H*Tlam/np.linalg.norm(Tlam+1e-6)
        Psi_T = do_stardt
        eta = -Psi_S.dot(Psi_T) + np.sqrt((Psi_S.dot(Psi_T))**2 + self.vr**2 - np.linalg.norm(Psi_T)**2)
        if np.isnan(eta):
            eta = 0
        Psi = eta*Psi_S + Psi_T
        self.Do = Do
        self.Psi_T = Psi_T
        return Psi

    def obstacle_callback(self, O):
        self.O = O
    
    def get_distance_vector(self):
        return self.Do
    
    def get_feedforward_vector(self):
        return self.Psi_T


# M = np.array([[0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 1, 0]])
class VectorField:

    def __init__(self,
                 parametric_curve, Ts=.1, vr=.25, kG=10, # curve field parameters
                 lam=0.25, M = np.array([[0, -1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]), vr_o=.25, kG_o=10, # obstacle field parameters
                 B_crit=0.05, B_safe=0.15, kB=20 # blending parameters
                 ):
        self.curve_vector_field = CurveVectorField(parametric_curve, Ts, vr, kG)
        self.obstacle_vector_field = ObstacleVectorField(lam=lam, M=M, vr=vr_o, kG=kG_o)
        self.Ts = Ts
        self.lam = lam # lambda parameter
        self.B_crit = B_crit
        self.B_safe = B_safe
        self.kB = kB

    def compute(self, p, t):
        # Compute fields
        curve = self.curve_vector_field.compute(p, t)
        obstacle = self.obstacle_vector_field.compute(p, t)
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
        print("\n--------------------")
        print("obst:", obstacle)
        print("curve", curve)
        print("Do", Do)
        print("|Do|", np.linalg.norm(Do))
        print("Theta", Theta, Theta1, Theta2)
        print("F", F)
        print('--------------------\n')
        return F


class Experiment:

    TAKEOFF_HEIGHT = 0.6
    TAKEOFF_DURATION = 2.5
    HOVER_DURATION = 1.0
    FLY_DRONE=True

    def __init__(self):
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cf_1 = self.swarm.allcfs.crazyflies[0]
        self.cf_2 = self.swarm.allcfs.crazyflies[1]

    def run(self):
        curve = lambda s,t: np.array([0.6*np.cos(-s), 0.6*np.sin(-s), 1.0+0*s+0.2*np.cos(0.2*t),
                                      0.6*np.cos(s), 0.6*np.sin(s), 1.0+0*s+0.2*np.cos(0.2*t)]).T
        vf = VectorField(curve)
        print("Taking off...")
        self.takeoff()
        try:
            print("Controlling...")
            while not rospy.is_shutdown():
                p_1 = self.cf_1.position()
                p_2 = self.cf_2.position()
                middle = (p_1 + p_2) / 2
                O = np.matrix(np.concatenate((middle, middle))).T
                vf.obstacle_vector_field.obstacle_callback(O)
                p = np.concatenate((p_1, p_2))
                t = self.timeHelper.time()
                v = vf.compute(p, t)
                v_1, v_2 = np.split(v, 2)
                if self.FLY_DRONE:
                    self.cf_1.cmdVelocityWorld(v_1, yawRate=0)
                    self.cf_2.cmdVelocityWorld(v_2, yawRate=0)
                else:
                    print(v)
                time.sleep(0.1)
        finally:
            print("Landing...")
            self.land()

    def takeoff(self):
        if self.FLY_DRONE:
            self.cf_1.takeoff(targetHeight=self.TAKEOFF_HEIGHT, duration=self.TAKEOFF_DURATION)
            self.cf_2.takeoff(targetHeight=self.TAKEOFF_HEIGHT, duration=self.TAKEOFF_DURATION)
        self.timeHelper.sleep(self.TAKEOFF_DURATION + self.HOVER_DURATION)

    def land(self):
        if self.FLY_DRONE:
            self.cf_1.land(targetHeight=0.04, duration=2.5)
            self.cf_2.land(targetHeight=0.04, duration=2.5)
            self.timeHelper.sleep(self.TAKEOFF_DURATION)
            self.cf_1.notifySetpointsStop()
            self.cf_2.notifySetpointsStop()


if __name__ == "__main__":
    # rospy.init_node('mysim')
    exp = Experiment()
    exp.run()

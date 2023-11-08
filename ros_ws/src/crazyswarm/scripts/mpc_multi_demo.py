#!/usr/bin/env python3

"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

import os
import numpy as np
from json import load
import pandas as pd

from VectorField import *
from MPC import *
from utils import *

# %% Setup Paths
# Keep track of the file path
file_path = os.path.dirname(os.path.realpath(__file__))
# Load experiment configs from /collision-avoidance/scripts
json_path = os.path.join(file_path, "config2.json")

# %% Load JSON
with open(json_path, "r", encoding="utf-8") as config_file:
    config = load(config_file)

    experiment_config = config["experiment"]
    agent_config = config["agent"]
    vector_field_config = config["vector_field"]
    mpc_config = config["mpc"]
    # collision_avoidance_config = config["collision_avoidance"]

    SIMULATION_TIME = config["experiment"]["SIMULATION_TIME"]
    SAMPLING_TIME = config["experiment"]["SAMPLING_TIME"]
    LANDING_HEIGHT = config["experiment"]["LANDING_HEIGHT"]
    COLLISION_AVOIDANCE_METHOD = experiment_config["COLLISION_AVOIDANCE_METHOD"]

    RADIUS = config["agent"]["RADIUS"]
    ID_LIST = config["agent"]["ID_LIST"]
    ID_LIST_SIZE = ID_LIST[-1] + 1

    KF = config["vector_field"]["KF"]
    VR = config["vector_field"]["VR"]

    arguments = config["vector_field"]["arguments"]
    curves = config["vector_field"]["curves"]

    # Model Predictive Control Parameters
    H = mpc_config["H"]
    ALPHA = mpc_config["ALPHA"]
    Q = np.diag(np.ones(6))
    R = np.diag(ALPHA*np.ones(3))
    RDU = np.diag(ALPHA*np.ones(3))
    A = np.array([  [1, 0, 0, SAMPLING_TIME, 0, 0], \
                    [0, 1, 0, 0, SAMPLING_TIME, 0], \
                    [0, 0, 1, 0, 0, SAMPLING_TIME], \
                    [0, 0, 0, 1, 0, 0], \
                    [0, 0, 0, 0, 1, 0], \
                    [0, 0, 0, 0, 0, 1]])
    B = np.array(   [[.5*SAMPLING_TIME**2, 0, 0], \
                    [0, .5*SAMPLING_TIME**2, 0], \
                    [0, 0, .5*SAMPLING_TIME**2], \
                    [SAMPLING_TIME, 0, 0], \
                    [0, SAMPLING_TIME, 0], \
                    [0, 0, SAMPLING_TIME]])
    TAKEOFF_DURATION = 5.0
    HOVER_DURATION = 5.0

# %% Import Crazyflie
# Then we can import Crazyswarm lib
from pycrazyswarm import Crazyswarm

DEBUG_VEL = False

def compute_state(last_state, LAST_STEP_TIME):
    current_state = np.zeros((ID_LIST_SIZE, 6))
    current_time = timeHelper.time()

    for id in ID_LIST:
        p = cfs[id].position()
        v = (p - last_state[id][:3])/LAST_STEP_TIME

        current_state[id][:3] = p
        current_state[id][3:] = v

        data.append([p[0], p[1], p[2], current_time-init_time, id, RADIUS])

    return current_state, current_time

def follow_field(id, state, time, vector_field):

    p = state[id][:3]
    v = state[id][3:]
    try:
        reference = computeReference(p, vector_field, H, time-init_time, SAMPLING_TIME)
    except Exception as error:
        print(error)
        return 0, p
    
    mpc.set_reference(reference)
    state = np.concatenate((p, v))
    
    try:
        u = mpc.step(state)
    except Exception as error:
        print(error)
        return 0, p

    cmd = v + u*SAMPLING_TIME

    if np.linalg.norm(p[0]) > 2:
        print("[SAFETY] Escaped X limmits.")
        print(p)
        return 0, p
    elif np.linalg.norm(p[1]) > 0.8:
        print("[SAFETY] Escaped Y limmits.")
        print(p)
        return 0, p
    elif p[2] > 2:
        print("[SAFETY] Escaped Z limmits.")
        print(p)
        return 0, p
    
    global DEBUG_VEL
    if not any(v):
        if DEBUG_VEL:
            print("DEBUG_VEL")
            return 0, p
        else:
            DEBUG_VEL = True

    cfs[id].cmdVelocityWorld(cmd, yawRate=0)
        
    return 1, p

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyfliesById

    # %% Vector Field
    # Set the curve associated by ID
    vector_fields = [None]*ID_LIST_SIZE
    for id in ID_LIST:
        curve = curves[id]
        # Build the Vector Field
        parametric_curve = eval(f"lambda {', '.join(arguments)}: {curve}")
        vector_fields[id] = VectorField(parametric_curve=parametric_curve,
                                simulation_time=SIMULATION_TIME,
                                Ts=SAMPLING_TIME,
                                vr=VR, Kf=KF)
    # %% Declaring Model Predictive Control

    mpc = MPC(6, 3, Q, R, RDU, H, SAMPLING_TIME, COLLISION_AVOIDANCE_METHOD)
    mpc.set_dynamics(A, B)

    # %% Declaring DataFrame
    data = []

    # %% Takeoff
    """
    
    Takeoff operation is needed due to ground effect.
    
    """
    swarm.allcfs.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    # Acquiring initial positions from all CFs
    init_time = timeHelper.time()
    state = np.zeros((ID_LIST_SIZE, 6))
    for id in ID_LIST:
        state[id][:3] = cfs[id].position()

    # Stabilizing all CFs at initial position after takeoff
    """
    
    After takeoff, agents usually have a non-zero velocity which may disturb the system at start.
    The following block seeks to override initial velocities, making the system start from rest conditions.

    Both goTo() and cmdPosition() methods where tested. The goTo() method work better considering the further
    application of the land() method.
    
    """
    while timeHelper.time() - init_time < HOVER_DURATION:
        for id in ID_LIST:
            cfs[id].goTo(state[id][:3], yaw=0, duration=HOVER_DURATION)
            # cfs[id].cmdPosition(state[id][:3]) # Tests indicate bad results before land()
        timeHelper.sleep(SAMPLING_TIME)

    # Establishing sample time for numerical derivative
    spent_time = 0
    LAST_STEP_TIME = SAMPLING_TIME
    
    """TO BE TESTED"""
    # while timeHelper.time() - init_time < SIMULATION_TIME:
    #     loop_start = timeHelper.time()

    #     state, current_time = compute_state(state, LAST_STEP_TIME)
    #     for id in ID_LIST:
    #         status = follow_field(id, state, current_time, vector_fields[id])
    #     initial_position = last_p
        
    #     loop_end = timeHelper.time()
    #     spent_time = loop_end - loop_start
    #     timeHelper.sleep(SAMPLING_TIME - spent_time)
    #     post_sleep = timeHelper.time()
    #     LAST_STEP_TIME = post_sleep - loop_start
    #     print(LAST_STEP_TIME, post_sleep - init_time, status)
    #     print()

    #     if not status:
    #         print("Exiting...")
    #         break

    # df = pd.DataFrame(data, columns=['x', 'y', 'z', 't', 'curve', 'mode'])
    # df.to_csv("experiment.csv", index=False)

    # kp = 0.1
    # ref = np.array([initial_position[0], initial_position[1], 0.0])
    # while cf.position()[2] > 0.2:
    #     print(cf.position()[2])
    #     print("Stabilizing before landing...")
    #     cf.cmdVelocityWorld(kp*(ref - cf.position()), yawRate=0)
    #     timeHelper.sleep(SAMPLING_TIME)

    # %% Landing
     """
     
     Tests indicated that landing command sometimes get passed by.
     Therefore, the following block will check the swarm distance to the ground after the landing order.
     If any agent fails to land, the order will be resent.
     
     Still in observation. Cases of ignored landing commands are not easily reproduced.
     
     """
    print("Landing...")
    while any(state[id][2] > 0.1 for id in ID_LIST):
        swarm.allcfs.land(targetHeight=0.04, duration=2.5)
        timeHelper.sleep(TAKEOFF_DURATION)
        state = np.zeros((ID_LIST_SIZE, 6))
        for id in ID_LIST:
            state[id][:3] = cfs[id].position()
    print("Landed.")

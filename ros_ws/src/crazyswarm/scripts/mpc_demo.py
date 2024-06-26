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
    TAKEOFF_DURATION = 2.0
    HOVER_DURATION = 2.0

# %% Import Crazyflie
# Then we can import Crazyswarm lib
from pycrazyswarm import Crazyswarm

DEBUG_VEL = False

def follow_field(initial_position, LAST_STEP_TIME):
    p = cf.position()
    current_time = timeHelper.time()
    v = (p - initial_position)/LAST_STEP_TIME
    # print(v, p, initial_position, LAST_STEP_TIME)

    data.append([p[0], p[1], p[2], current_time-init_time, cf.id, 0.05])

    try:
        reference = computeReference(p, vector_field, H, current_time-init_time, SAMPLING_TIME)
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
    
    # phi = vector_field.compute(p, current_time)

    cmd = v + u*SAMPLING_TIME

    # print(phi, reference[0])
    # print(cmd, u)

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

    cf.cmdVelocityWorld(cmd, yawRate=0)
        
    return 1, p

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # %% Vector Field
    # Set the curve associated by ID
    curve = curves[5]
    # Build the Vector Field
    parametric_curve = eval(f"lambda {', '.join(arguments)}: {curve}")
    vector_field = VectorField(parametric_curve=parametric_curve,
                               simulation_time=SIMULATION_TIME,
                               Ts=SAMPLING_TIME,
                               vr=VR, Kf=KF)

    # %% Declaring Model Predictive Control

    mpc = MPC(6, 3, Q, R, RDU, H, SAMPLING_TIME, COLLISION_AVOIDANCE_METHOD)
    mpc.set_dynamics(A, B)

    # %% Declaring DataFrame
    data = []

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    init_time = timeHelper.time()
    initial_position = cf.position()

    while timeHelper.time() - init_time < HOVER_DURATION:
        cf.cmdPosition(initial_position)
        timeHelper.sleep(SAMPLING_TIME)


    spent_time = 0
    LAST_STEP_TIME = SAMPLING_TIME
    
    while timeHelper.time() - init_time < SIMULATION_TIME:
        loop_start = timeHelper.time()

        status, last_p = follow_field(initial_position, LAST_STEP_TIME)
        initial_position = last_p
        
        loop_end = timeHelper.time()
        spent_time = loop_end - loop_start
        timeHelper.sleep(SAMPLING_TIME - spent_time)
        post_sleep = timeHelper.time()
        LAST_STEP_TIME = post_sleep - loop_start
        print(LAST_STEP_TIME, post_sleep - init_time, status)
        print()

        if not status:
            print("Exiting...")
            break

    df = pd.DataFrame(data, columns=['x', 'y', 'z', 't', 'curve', 'mode'])
    df.to_csv("experiment.csv", index=False)

    kp = 0.1
    ref = np.array([initial_position[0], initial_position[1], 0.0])
    while cf.position()[2] > 0.2:
        print(cf.position()[2])
        print("Stabilizing before landing...")
        cf.cmdVelocityWorld(kp*(ref - cf.position()), yawRate=0)
        timeHelper.sleep(SAMPLING_TIME)

    # while timeHelper.time() - init_time < post_sleep - init_time + HOVER_DURATION:
    #     print("Stabilizing before landing...")
    #     cf.cmdPosition(initial_position)
    #     timeHelper.sleep(SAMPLING_TIME)

    print("Landing...")
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)
    print("Landed.")

    # initial_position[2] = 0.0
    # cf.goTo(initial_position, 0, HOVER_DURATION)
    # timeHelper.sleep(HOVER_DURATION)

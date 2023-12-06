#!/usr/bin/env python3

"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

import sys
import os
import numpy as np
from json import load
import time
import pandas as pd

from VectorField import *
from MPC import *
from utils import *

# %% Setup Paths
# Keep track of the file path
file_path = os.path.dirname(os.path.realpath(__file__))
# Load experiment configs from /collision-avoidance/scripts
json_path = os.path.join(file_path, "mpc_ca_demo.json")

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
    TAKEOFF_DURATION = 2.5
    HOVER_DURATION = 5.0

# %% Import Crazyflie
# Then we can import Crazyswarm lib
from pycrazyswarm import Crazyswarm

def follow_field():
    try:
        current_time = timeHelper.time()
        
        p = cf.position()
        v = vector_field.compute(p, current_time-init_time)
        cf.cmdVelocityWorld(v, yawRate=0)

        data.append([p[0], p[1], p[2], current_time-init_time, 3, RADIUS])
        cmd_data.append([v[0], v[1], v[2], current_time-init_time, 3])
        
        return 1
    except Exception as error:
        print(error)
        return 0

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    data = []
    cmd_data = []

    # %% Vector Field
    # Set the curve associated by ID
    curve = curves[5]
    # Build the Vector Field
    parametric_curve = eval(f"lambda {', '.join(arguments)}: {curve}")
    vector_field = VectorField(parametric_curve=parametric_curve,
                               simulation_time=SIMULATION_TIME,
                               Ts=SAMPLING_TIME,
                               vr=VR, Kf=KF)

    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    init_time = timeHelper.time()
    
    while timeHelper.time() - init_time < SIMULATION_TIME:
        loop_start = timeHelper.time()
        status = follow_field()
        if not status:
            break
        loop_end = timeHelper.time()
        spent_time = loop_end - loop_start
        timeHelper.sleep(SAMPLING_TIME - spent_time)


    df = pd.DataFrame(data, columns=['x', 'y', 'z', 't', 'curve', 'mode'])
    df.to_csv("experiment.csv", index=False)
    cmd_df = pd.DataFrame(cmd_data, columns=['vx', 'vy', 'vz', 't', 'id'])
    cmd_df.to_csv("experiment_u.csv", index=False)

    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)

#!/usr/bin/env python3

"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

import os
import numpy as np
from json import load
import pandas as pd

from CAMPC.VectorField import *
from CAMPC.MPC import *
from CAMPC.CollisionAvoidance import *
from CAMPC.utils import *

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
    ID_LIST = config["agent"]["ID_LIST"]
    ID_LIST_SIZE = ID_LIST[-1] + 1

    KF = config["vector_field"]["KF"]
    VR = config["vector_field"]["VR"]

    arguments = config["vector_field"]["arguments"]
    curves = config["vector_field"]["curves"]

    # Model Predictive Control Parameters
    H = mpc_config["H"]
    ALPHA = mpc_config["ALPHA"]
    BETA = mpc_config["BETA"]
    SLACK = mpc_config["SLACK"]
    Q = np.diag(np.ones(6))
    Q[2][2] = 2
    Q[5][5] = 2
    print(Q)
    R = np.diag(ALPHA*np.ones(3))
    RDU = np.diag(BETA*np.ones(3))
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
    # Collision Avoidance Parameters
    MAX_ACC = config["collision_avoidance"]["MAX_ACC"]
    GAMMA = config["collision_avoidance"]["GAMMA"]

# %% Import Crazyflie
# Then we can import Crazyswarm lib
from pycrazyswarm import Crazyswarm

DEBUG_VEL = np.zeros(ID_LIST_SIZE)

def compute_state(last_state, LAST_STEP_TIME):
    global data

    current_state = np.zeros((ID_LIST_SIZE, 6))
    current_time = timeHelper.time()

    for id in ID_LIST:
        p = cfs[id].position()
        v = (p - last_state[id][:3])/LAST_STEP_TIME

        current_state[id][:3] = p
        current_state[id][3:] = v

        agents[id].position = p
        agents[id].velocity = v

        data.append([p[0], p[1], p[2], current_time-init_time, id, RADIUS])

    return current_state, current_time

def follow_field(id, state, agents, time, vector_field):
    global cmd_data

    p = state[id][:3]
    v = state[id][3:]

    # %% Safety Checkup
    # Checks if inside the safe zone
    if np.linalg.norm(p[0]) > 2.2:
        print("[SAFETY] Escaped X limmits.")
        return 0
    elif np.linalg.norm(p[1]) > 1.2:
        print("[SAFETY] Escaped Y limmits.")
        return 0
    elif p[2] > 2:
        print("[SAFETY] Escaped Z limmits.")
        return 0
    # Checks if anybody stopped working
    global DEBUG_VEL
    if all(np.linalg.norm(v[i]) < 1e-4 for i in range(len(v))):
        if DEBUG_VEL[id] > 10:
            print("DEBUG_VEL")
            return 0
        else:
            DEBUG_VEL[id] += 1
    else:
        DEBUG_VEL[id] = 0
    
    try:
        reference = computeReference(p, vector_field, H, time-init_time, SAMPLING_TIME)
    except Exception as error:
        print(error)
        return 0
    
    mpc.set_reference(reference)
    state = np.concatenate((p, v))

    if COLLISION_AVOIDANCE_METHOD == 'cbf':
        A_cbf, b_cbf = collision_avoidance.compute_group_barrier_function(agents[id], agents)
        parameters = {
            "A" : A_cbf,
            "b" : b_cbf
        }
    elif COLLISION_AVOIDANCE_METHOD == 'orca':
        orca_u, orca_n, propagated_states = collision_avoidance.compute_group_orca(agents[id], agents, tau=5, dt=SAMPLING_TIME)
        parameters = {
            "orca_u" : orca_u,
            "orca_n" : orca_n,
            "propagated_states" : propagated_states
        }
    else:
        parameters = None
    mpc.set_collision_avoidance_parameters(parameters)
    
    try:
        u = mpc.step(state)
    except Exception as error:
        print(error)
        return 0

    cmd = v + u*SAMPLING_TIME

    # if np.linalg.norm(cmd) < .05:
    #     print(cmd, time)

    """DEBUG CMD"""
    cfs[id].cmdVelocityWorld(cmd, yawRate=0)

    cmd_data.append([v[0], v[1], v[2], u[0], u[1], u[2], current_time-init_time, id])
        
    return 1

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyfliesById

    # %% Declaring agents
    """
    Collision Avoidance methods were developed based on the Agent() class.
    It needs a rework due its non-generalistic structure.

    """
    agents = []
    for id in range(ID_LIST_SIZE):
        agents.append(Agent(id=id, p=np.array([0., 0., 0.]), v=np.array([0., 0., 0.]), a=np.array([0., 0., 0.]), r=RADIUS, Ts=SAMPLING_TIME))

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

    mpc = MPC(6, 3, Q, R, RDU, H, SAMPLING_TIME, COLLISION_AVOIDANCE_METHOD, SLACK)
    mpc.set_dynamics(A, B)

    # %% Declaring Collision Avoidance

    collision_avoidance = CollisionAvoidance(H, RADIUS, MAX_ACC, GAMMA)
    collision_avoidance.set_agent_dynamic(A)

    # %% Declaring DataFrame
    data = []
    cmd_data = []

    # %% Takeoff
    """
    
    Takeoff operation is needed due to ground effect.
    
    """
    """DEBUG CMD"""
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
            """DEBUG CMD"""
            cfs[id].goTo(state[id][:3], yaw=0, duration=HOVER_DURATION)
        timeHelper.sleep(SAMPLING_TIME)

    # Establishing sample time for numerical derivative
    spent_time = 0
    LAST_STEP_TIME = SAMPLING_TIME
    status = np.array([None]*ID_LIST_SIZE)
    
    """TO BE TESTED"""
    while timeHelper.time() - init_time < SIMULATION_TIME:
        loop_start = timeHelper.time()

        state, current_time = compute_state(state, LAST_STEP_TIME)
        for id in ID_LIST:
            status[id] = follow_field(id, state, agents, current_time, vector_fields[id])
        
        loop_end = timeHelper.time()
        spent_time = loop_end - loop_start
        timeHelper.sleep(SAMPLING_TIME - spent_time)
        post_sleep = timeHelper.time()
        LAST_STEP_TIME = post_sleep - loop_start

        if not all(status[ID_LIST]):
            print("Emergency Stoppage.")
            swarm.allcfs.emergency()
            df = pd.DataFrame(data, columns=['x', 'y', 'z', 't', 'curve', 'mode'])
            df.to_csv("experiment.csv", index=False)
            cmd_df = pd.DataFrame(cmd_data, columns=['vx', 'vy', 'vz', 'ax', 'ay', 'az', 't', 'id'])
            cmd_df.to_csv("experiment_u.csv", index=False)
            quit()


    df = pd.DataFrame(data, columns=['x', 'y', 'z', 't', 'curve', 'mode'])
    df.to_csv("experiment.csv", index=False)
    cmd_df = pd.DataFrame(cmd_data, columns=['vx', 'vy', 'vz', 'ax', 'ay', 'az', 't', 'id'])
    cmd_df.to_csv("experiment_u.csv", index=False)

    # %% Landing
    """
    
    Tests indicated that landing command sometimes get passed by.
    Therefore, the following block will check the swarm distance to the ground after the landing order.
    If any agent fails to land, the order will be resent.
    
    Still in observation. Cases of ignored landing commands are not easily reproduced.
    
    """
    print("Landing...")

    kp = 0.1
    landing_sites = [[state[id][0], state[id][1], 0.3] for id in range(ID_LIST_SIZE)]
    land_flag = [False]*ID_LIST_SIZE
    while any(state[id][2] > 0.1 for id in ID_LIST):
        for id in ID_LIST:
            if state[id][2] > 0.4:
                """DEBUG CMD"""
                cfs[id].cmdVelocityWorld(kp*(landing_sites[id] - state[id][:3]), yawRate=0)
            elif state[id][2] > 0.1 and not land_flag[id]:
                """DEBUG CMD"""
                cfs[id].land(targetHeight=0.04, duration=2.5)
                land_flag[id] = True
                print("Agent " + str(id) + " Landed.")
        timeHelper.sleep(SAMPLING_TIME)
        state = np.zeros((ID_LIST_SIZE, 6))
        for id in ID_LIST:
            state[id][:3] = cfs[id].position()
    print("All Landed.")

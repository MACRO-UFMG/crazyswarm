import pandas as pd
import numpy as np

from pycrazyswarm import *

## my libs
from online_path_planning_denurbs.scripts.__utils import *
from online_path_planning_denurbs.scripts.__utils_yaml import *
from online_path_planning_denurbs.scripts.nurbs import *
from online_path_planning_denurbs.scripts.control import *
from online_path_planning_denurbs.scripts.robot_models import *
from online_path_planning_denurbs.scripts.view_experiment import *


INPUT_YAML_FILE_NAME = "online_path_planning_denurbs/config/config_1static_obs.yaml"
yaml_utils = YAML_utils(filename=INPUT_YAML_FILE_NAME)

TAKEOFF_DURATION    = 2
SIMULATION_TIME     = 60
SAMPLING_TIME       = 0.1
DEGREE              = 7
NUM_SAMPLES         = 300
KF_FIELD            = 2.0

ALTITUDE = 1.

s_delta = 0.05
s_f = 0.3
s_i = 0.0

""" robot parameters"""
vrobot = 0.2
vobs = 0.3
deltaVel = 0.1

vr = vrobot + deltaVel
r_control = 1/KF_FIELD*np.tan(np.pi/2*deltaVel/vr)


r = 0.0
pmin = 0.1


Rsafe = 0.2 + r_control 
Rview = 1.2


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    data = []

    # TAKEOFF
    cf.takeoff(targetHeight=ALTITUDE, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    previous_position = cf.position()
    target_position = [1.5, 1.0]

    init_time = timeHelper.time()
    
    while timeHelper.time() - init_time < SIMULATION_TIME:
        loop_start = timeHelper.time()

        # Compute Reference
        ###################
        """ Setting for the curve """
        ctrl_points, weights = yaml_utils.getControlWeightsfromConfig(yaml_utils.read_yaml())
        knot = yaml_utils.read_yaml()["knotvector"]
        evalpts_x, evalpts_y, curve = pathNURBS.nurbs(degree=DEGREE, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot)

        """ Update robot using the vector field control """
        curve.delta = 1/15000
        eval_func = curve.evaluate_single

        current_position = cf.position()

        if np.linalg.norm(current_position[0:2] - target_position) < 0.1:
            break
        elif current_position[0] > 1.5 or current_position[1] > 1:
            break

        field, s_star = compute_field(eval_func, pos=current_position[0:2], s_i = s_i, s_f = s_f, Kf=KF_FIELD, vr=vrobot)
        curve.delta = 1/NUM_SAMPLES

        if s_star - s_delta <= 0:
            s_i = 0
            s_f = 1
            # print('aqui 0')
        elif s_star + s_delta >= 1:
            s_i = 0
            s_f = 1
            # print('aqui 1')
        else:
            s_i = s_star - s_delta
            s_f = s_star + s_delta
        ###################
        # Apply Command
        ###################
        """ send control (vel) to the robot """
        kp = 0.1
        altitude_control = kp*(ALTITUDE - current_position[2])

        v = np.array([field[0], field[1], altitude_control])
        cf.cmdVelocityWorld(v, yawRate=0)
        ###################
        
        loop_end = timeHelper.time()
        spent_time = loop_end - loop_start
        timeHelper.sleep(SAMPLING_TIME - spent_time)

        ###################
        # RECORD DATA
        ###################

        data.append([current_position[0], current_position[1],
                     (current_position[0] - previous_position[0])/SAMPLING_TIME, (current_position[1] - previous_position[1])/SAMPLING_TIME,
                     v[0], v[1],
                     loop_end - init_time, 1
        ])
        previous_position = current_position

        ###################


    df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy', 'ux', 'uy', 't', 'id'])
    df.to_csv("experiment.csv", index=False)

    cf.cmdVelocityWorld([0., 0., 0.], yawRate=0)
import pandas as pd
import numpy as np
import pickle

from pycrazyswarm import *

## my libs
from online_path_planning_denurbs.scripts.__utils import *
from online_path_planning_denurbs.scripts.__utils_yaml import *
from online_path_planning_denurbs.scripts.nurbs import *
from online_path_planning_denurbs.scripts.control import *
from online_path_planning_denurbs.scripts.robot_models import *
from online_path_planning_denurbs.scripts.view_experiment import *



AGENT_YAML_FILE_NAME = "online_path_planning_denurbs/config/config_agent.yaml"
PLANNER_YAML_FILE_NAME = "online_path_planning_denurbs/config/config_planner.yaml"
OPT_YAML_FILE_NAME = "online_path_planning_denurbs/config/config_opt.yaml"

yaml_agent = YAML_utils(filename=AGENT_YAML_FILE_NAME)
yaml_opt = YAML_utils(filename=OPT_YAML_FILE_NAME)
yaml_planner = YAML_utils(filename=PLANNER_YAML_FILE_NAME)

yaml_planner.write_parameters_onFile(start=0)

NAME_EXPERIMENT = "1obs"

TAKEOFF_DURATION    = 2
SIMULATION_TIME     = 35
SAMPLING_TIME       = 0.1
DEGREE              = 7
NUM_SAMPLES         = 450
KF_FIELD            = 6.0
KVEL                = 1.18 #para vrobot = 0.3
TARGET_POSITION = np.array(yaml_planner.read_yaml()["target_position"], dtype=float)

ALTITUDE = 1.

s_delta = 0.05
s_f = 0.3
s_i = 0.0

""" robot parameters"""
vrobot = 0.3

deltaVel = 0.1

# vr = vrobot - deltaVel
# r_control = 1/KF_FIELD*np.tan((np.pi/2)*deltaVel/vr)
r_control = 0.121

r = 0.0
pmin = 0.25


Rsafe = 0.15 + r_control + 0.075
Rview = 1.55
print(f'Rsafe = {Rsafe}, rcontrol = {r_control}')

pobs0 = np.array([-1.3, 0.0, 0.0, 0.5+r_control+0.075])
vobs0 = np.array([0.0, 0.0, 0.0])

pobs1 = np.array([1.1, -1.1, 0.0, Rsafe])
vdir1 = np.array([-0, 0.5, 0.0]) - pobs1[:3]
vobs1 = 0.09*vdir1/np.linalg.norm(vdir1)

pobs2 = np.array([1.,0.5, 0.0, Rsafe])
vdir2 = np.array([-1, -0.5, 0.0]) - pobs2[:3]
vobs2 = 0.09*vdir2/np.linalg.norm(vdir2)

pobs3 = np.array([0, 1., 0.0, Rsafe])
vdir3 = np.array([-2.5, -1, 0.0]) - pobs3[:3]
vobs3 = 0.08*vdir3/np.linalg.norm(vdir3)

pobs4 = np.array([-1, -2, 0.0, Rsafe])
vdir4 = np.array([0, 1, 0.0]) - pobs4[:3]
vobs4 = 0.1*vdir4/np.linalg.norm(vdir4)

# pobs0[:2] *= 100
pobs1[:2] *= 100
pobs3[:2] *= 100
pobs4[:2] *= 100
pobs2[:2] *= 100

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    data = []
    list_curve_x = []
    list_curve_y = []
    list_pos_agent = []
    list_pos_obstacles = []
    list_time = []
    list_projected_point = []
    list_detected_obstacles = []
    # TAKEOFF
    cf.takeoff(targetHeight=ALTITUDE, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    
    previous_position = cf.position()
    start = 0
    # yaml_planner.write_parameters_onFile(start=1)

    init_time = timeHelper.time()

    config = yaml_agent.read_yaml()
    ctrl_points = np.array(config["control_points"], dtype=float)
    weights = np.array(config["weights"], dtype=float)
    knot = np.array(config["knotvector"], dtype=float)
    
    """ saving new curve to the planner """
    # yaml_planner.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights=weights.tolist(), knotvector=knot.tolist())
    
    evalpts_x, evalpts_y, curve = pathNURBS.nurbs(degree=DEGREE, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot.tolist())


    list_time.append(0)
    list_pos_agent.append(previous_position)
    list_pos_obstacles.append([pobs0.copy(), pobs1.copy(), pobs2.copy(), pobs3.copy(), pobs4.copy()])
    
    list_curve_x.append(evalpts_x)
    list_curve_y.append(evalpts_y)
    projected_point = np.array(yaml_agent.read_yaml()["projected_point"], dtype=float)
    list_projected_point.append(projected_point)
    list_detected_obstacles.append(1)
    
    try:
        while timeHelper.time() - init_time < SIMULATION_TIME:
            loop_start = timeHelper.time()            


            # Compute Reference
            ###################
            """ Setting for the curve """
            # ctrl_points, weights = yaml_utils.getControlWeightsfromConfig(yaml_utils.read_yaml())
            config = yaml_agent.read_yaml()
            ctrl_points = np.array(config["control_points"], dtype=float)
            weights = np.array(config["weights"], dtype=float)
            knot = np.array(config["knotvector"], dtype=float)
            
            """ saving new curve to the planner """
            # yaml_planner.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights=weights.tolist(), knotvector=knot.tolist())
            
            evalpts_x, evalpts_y, curve = pathNURBS.nurbs(degree=DEGREE, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot.tolist())

            """ Update robot using the vector field control """
            curve.delta = 1/15000
            eval_func = curve.evaluate_single

            current_position = cf.position()
            pos = current_position.copy()
            # print(f'pos robo = {pos}')
            """ Update position and vel to the planner """
            # yaml_planner.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights=weights.tolist(), knotvector=knot.tolist(),\
                                                #  agent_pos=pos.tolist(), agent_vel=((current_position-previous_position)/SAMPLING_TIME).tolist())

            detected_obs = [pobs0.copy()]
            detected_vobs = [vobs0.copy()]
            # print(np.linalg.norm(current_position - pobs1[:3]))
            if np.linalg.norm(current_position[:2] - pobs1[:2]) <= Rview:
                detected_obs.append(pobs1.copy())
                detected_vobs.append(vobs1.copy())
                # print('detected obs1')
            if np.linalg.norm(current_position[:2] - pobs2[:2]) <= Rview:
                # print('detected obs1')
                detected_obs.append(pobs2.copy())
                detected_vobs.append(vobs2.copy())
            if np.linalg.norm(current_position[:2] - pobs3[:2]) <= Rview:
                detected_obs.append(pobs3.copy())
                detected_vobs.append(vobs3.copy())
                # print('detected obs1')
            if np.linalg.norm(current_position[:2] - pobs4[:2]) <= Rview:
                # print('detected obs1')
                detected_obs.append(pobs4.copy())
                detected_vobs.append(vobs4.copy())

            list_pobs = np.array(detected_obs)
            list_vobs = np.array(detected_vobs)
            list_detected_obstacles.append(len(detected_obs))
            yaml_planner.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights=weights.tolist(), knotvector=knot.tolist(),\
                                                agent_pos=pos.tolist(), agent_vel=((current_position-previous_position)/SAMPLING_TIME).tolist(),\
                                                obs=list_pobs.tolist(), vobs=list_vobs.tolist())

            # yaml_planner.write_parameters_onFile(obs=list_pobs.tolist(), vobs=list_vobs.tolist())
            
            if np.linalg.norm(current_position[:2] - TARGET_POSITION[:2]) < 0.1:
                print(np.linalg.norm(current_position[:2] - TARGET_POSITION[:2]))
                break
            # elif current_position[0] > 1.5 or current_position[1] > 1:
            #     break

            field, s_star = compute_field(eval_func, pos=current_position[0:2], s_i = s_i, s_f = s_f, Kf=KF_FIELD, vr=vrobot)
            curve.delta = 1/NUM_SAMPLES
            # print(f' vel = {np.linalg.norm(field)}')

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
            
            v = np.array([field[0]*KVEL, field[1]*KVEL, altitude_control])
            cf.cmdVelocityWorld(v, yawRate=0)
            ###################
            
            loop_end = timeHelper.time()
            spent_time = loop_end - loop_start
            if(spent_time < SAMPLING_TIME):
                timeHelper.sleep(SAMPLING_TIME - spent_time)
            loop_end = timeHelper.time()
            spent_time = loop_end - loop_start
            # print(f'spent_time = {spent_time:.2f}')
            ###################
            # RECORD DATA
            ###################
            vel = (current_position[0] - previous_position[0])/SAMPLING_TIME
            if start == 0 and np.linalg.norm(vel) > 0.2:
                yaml_planner.write_parameters_onFile(start=1)
                start = 1

            data.append([current_position[0], current_position[1],
                        (current_position[0] - previous_position[0])/SAMPLING_TIME, (current_position[1] - previous_position[1])/SAMPLING_TIME,
                        v[0], v[1],
                        loop_end - init_time, 1
            ])
            data.append([pobs1[0], pobs1[1],
                        0.0, 0.0,
                        vobs1[0], vobs1[1],
                        loop_end - init_time, 2
            ])
            data.append([pobs2[0], pobs2[1],
                        0.0, 0.0,
                        vobs2[0], vobs2[1],
                        loop_end - init_time, 3
            ])

            list_time.append(timeHelper.time() - init_time)
            list_pos_agent.append(current_position)
            list_pos_obstacles.append([pobs0.copy(), pobs1.copy(), pobs2.copy(), pobs3.copy(), pobs4.copy()])
            
            list_curve_x.append(evalpts_x)
            list_curve_y.append(evalpts_y)
            projected_point = np.array(yaml_agent.read_yaml()["projected_point"], dtype=float)
            list_projected_point.append(projected_point)


            pobs1[:2] = pobs1[:2] + vobs1[:2]*SAMPLING_TIME
            pobs2[:2] = pobs2[:2] + vobs2[:2]*SAMPLING_TIME
            pobs3[:2] = pobs3[:2] + vobs3[:2]*SAMPLING_TIME
            pobs4[:2] = pobs4[:2] + vobs4[:2]*SAMPLING_TIME

            # print(list_pos_obstacles[-1][1])
            # print(pobs2[:2] - current_position[:2])
            previous_position = current_position

            if np.linalg.norm(current_position[:2] - TARGET_POSITION[:2]) < 0.1:
                break

            ###################

    except Exception as e:
        print(e)
        df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy', 'ux', 'uy', 't', 'id'])
        df.to_csv("experiment.csv", index=False)
        cf.cmdVelocityWorld([0., 0., 0.], yawRate=0)

    df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy', 'ux', 'uy', 't', 'id'])
    df.to_csv("experiment_"+NAME_EXPERIMENT+".csv", index=False)

    parameters = ['t_sim', "list_curve_x", "list_curve_y", "list_pos_agent", "list_pos_obstacles","list_projected_point","list_detected_obstacles",
            ]
    dict_parameters = dict(zip(parameters, range(len(parameters))))
    with open("experiment_"+NAME_EXPERIMENT+".pkl", 'wb') as f: 
        pickle.dump([list_time, list_curve_x, list_curve_y, list_pos_agent, list_pos_obstacles,list_projected_point,list_detected_obstacles,
                    dict_parameters], f)


    yaml_planner.write_parameters_onFile(start=0)

    for _ in range(20):
        print("LANDING: Started...")
        ref = 0.3 - cf.position()[2]
        cf.cmdVelocityWorld([0., 0., 0.1*ref], yawRate=0)
        timeHelper.sleep(0.2)
    
    print("LANDING: Finished.")
#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import time
import re
import subprocess
import imageio
import csv
import pickle
from datetime import datetime


## my libs
from online_path_planning_denurbs.scripts.__utils import *
from online_path_planning_denurbs.scripts.__utils_yaml import *
from online_path_planning_denurbs.scripts.nurbs import *
from online_path_planning_denurbs.scripts.control import *
from online_path_planning_denurbs.scripts.robot_models import *
from online_path_planning_denurbs.scripts.view_experiment import *


INPUT_YAML_FILE_NAME = "online_path_planning_denurbs/config/config_1static_obs_copy.yaml"
yaml_utils = YAML_utils(filename=INPUT_YAML_FILE_NAME)



def runplanner(VO=True, verbose=False):
    """ Run the path planner DE3D-NURBS (a c++ object) """
    if VO:
        running = "./online_path_planning_denurbs/source/main_VO"
    else:
        running = "./online_path_planning_denurbs/source/main"
    if verbose:
        subprocess.run([running, "1", INPUT_YAML_FILE_NAME])
    else:
        subprocess.run([running, "1", INPUT_YAML_FILE_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 
   
    new_ctrl_points, weights = yaml_utils.getControlWeightsfromConfig(yaml_utils.read_yaml())

    return new_ctrl_points, weights




""" Parameters of the experiment and optimization problem """


"""experiment parameters """
PLANNING_TIME = 0.4
SAVE_EXPERIMENT = True
PATH_RESULTS = "./online_path_planning_denurbs/results/"
NAGENTS = 0
TARGET_POSITION = np.array(yaml_utils.read_yaml()["target_position"])[0][:2]
print(TARGET_POSITION)
DEBUG = False
VO_flag = True

""" robot parameters"""
vrobot = 0.2
deltaVel = 0.1


""" optimization parameters """
tVO = 1.2*PLANNING_TIME
degree = 7
NUM_SAMPLES = 400
n_other_opm_var = 1
nctrl_notchange = 3
scale = 2

# li = [-1.*scale,-1.*scale,-1.*scale,-1.*scale,-1.*scale,-1.*scale,-1.*scale,-1.*scale,-1.00001,-1.00001,-1.00001,-1.00001,-1,-1, -deltaVel]
# ui = [1.*scale,1.*scale,1.*scale,1.*scale,1.*scale,1.*scale,1.*scale,1.*scale,1.0,1.0,1.0,1.0,0.00,0.00, deltaVel]
yaml_utils.write_parameters_onFile(tVO=tVO, aval=18000, npi=int(18*15), tsampling=PLANNING_TIME, degree=degree)

""" Running simulation """
""" List of detected obstacles """
list_detected_obstacles = yaml_utils.read_yaml()["obs"]
list_detected_vobs = yaml_utils.read_yaml()["vobs"]
obstacles = []
vel_obs = []
for obs, vobs in zip(list_detected_obstacles, list_detected_vobs):
    """ projection of the obstacles"""
    obstacles.append([obs[0]+vobs[0]*PLANNING_TIME,obs[1]+vobs[1]*PLANNING_TIME, obs[2]+vobs[2]*PLANNING_TIME, obs[3]])
    vel_obs.append([vobs[0], vobs[1], vobs[1]])
obstacles = np.array(obstacles)
vel_obs = np.array(vel_obs)


""" initial path """
new_ctrl_points = yaml_utils.read_yaml()["control_points"]
new_weights = yaml_utils.read_yaml()["weights"]
new_vrobot = yaml_utils.read_yaml()["vcurve"]
new_knotvector = yaml_utils.read_yaml()["knotvector"]
ctrl_points = new_ctrl_points.copy()
weights = new_weights.copy()
knot = new_knotvector.copy()
evalpts_x, evalpts_y, curve = pathNURBS.nurbs(degree=degree, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot)

previous_target_point = ctrl_points[0]


li = [[-1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[-1.00001]*(len(weights)-nctrl_notchange*2),-1,-1, -deltaVel]
li = flatten_list(li)
ui = [[1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[1.0]*(len(weights)-nctrl_notchange*2), 0.00,0.00, deltaVel]
ui = flatten_list(ui)
yaml_utils.write_parameters_onFile(li=li, ui=ui)


""" Main loop """
t_sim = 0
t_run = []
t_list = []
ctrl_points_list = []
weights_list = []
knot_list = []
curve_list = []

agent_pos = yaml_utils.read_yaml()["agent_pos"]
agent_vel = yaml_utils.read_yaml()["agent_vel"]


while (np.linalg.norm(agent_pos[:2] - TARGET_POSITION) >= .1):   
    
    
        
    tt = time.time()
    

    # """   save old curve  """
    # old_evalpts_x, old_evalpts_y, old_curve = pathNURBS.nurbs(degree=degree, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot)

    """ update to a new cruise speed"""
    vrobot = new_vrobot

    """  update new curve """
    print(f"\nCurva atualizada em t = {t_sim:.2f}")
    ctrl_points = new_ctrl_points.copy()
    weights = new_weights.copy()
    knot = new_knotvector.copy()
    evalpts_x, evalpts_y, curve = pathNURBS.nurbs(degree=degree, points=ctrl_points, weigths=weights, dt=1/NUM_SAMPLES, knot=knot)

    

    
    
    """ cut the curve in the project future position"""    
    _, _, curve_to_cut = pathNURBS.nurbs(degree=degree, points=ctrl_points.copy(), weigths=weights.copy(), dt=1/NUM_SAMPLES, knot=knot.copy())
    dist_robot2goal = np.linalg.norm(agent_pos[:2] - TARGET_POSITION)
    if dist_robot2goal <= 1.0:
        Nsamples = 400
    elif dist_robot2goal <= 0.5:
        Nsamples = 200
    else:
        Nsamples = 1500
    target_point_to_planner = project_future_position_on_curve(curve_to_cut, 0 , PLANNING_TIME, 0, vrobot, None, follow_field=False, N=Nsamples)
    previous_target_point = target_point_to_planner
    
    
    projected_ctrl_points, projected_weights, new_knotvector_after_split = pathNURBS.split_closest_point_2d(curve_to_cut,target_point_to_planner, take_before=0.0) 
    
        
    # if len(projected_weights) != len(weights):
    #     # yaml_utils.write_parameters_onFile(d = (len(projected_weights)-6)*3+2+n_other_opm_var)
    #     #yaml_utils.read_yaml()["d"] - (len(weights) - len(projected_weights))
    #     print(f"change d, len = {len(projected_weights)}")      
    
    
    evalpts_projected_x, evalpts_projected_y, _ = pathNURBS.nurbs(degree=degree, points=projected_ctrl_points, weigths=projected_weights, dt=1/NUM_SAMPLES,knot=new_knotvector_after_split)
    cutted_evalpts_x, cutted_evalpts_y, _ = pathNURBS.nurbs(degree=degree, points=projected_ctrl_points, weigths=projected_weights, dt=1/NUM_SAMPLES, knot=new_knotvector_after_split)

    # projected_ctrl_points, projected_weights = ctrl_points.copy(), weights.copy()
    # target_point_to_planner = ctrl_points[0]

    """ Update the list of detected obstacles """
    list_detected_obstacles = yaml_utils.read_yaml()["obs"]
    list_detected_vobs = yaml_utils.read_yaml()["vobs"]
    obstacles = []
    vel_obs = []
    for obs, vobs in zip(list_detected_obstacles, list_detected_vobs):
        """ projection of the obstacles"""
        obstacles.append([obs[0]+vobs[0]*PLANNING_TIME,obs[1]+vobs[1]*PLANNING_TIME, obs[2]+vobs[2]*PLANNING_TIME, obs[3]])
        vel_obs.append([vobs[0], vobs[1], vobs[1]])
    obstacles = np.array(obstacles)
    vel_obs = np.array(vel_obs)
    

    """ set new parameters to planner"""
    if np.linalg.norm(np.array([evalpts_x[0], evalpts_y[0]]) - TARGET_POSITION) < 1:
        scale = 1.5            
    elif np.linalg.norm(np.array([evalpts_x[0], evalpts_y[0]]) - TARGET_POSITION) <= 0.6:
        scale = 0.6           
    elif np.linalg.norm(np.array([evalpts_x[0], evalpts_y[0]]) - TARGET_POSITION) <= 0.5:
        scale = 0.5            
    else:
        scale = 2
    li = [[-1.0*scale]*((len(projected_weights)-nctrl_notchange*2)*2),[-1.00001]*(len(projected_weights)-nctrl_notchange*2),-1,-1, -deltaVel]
    li = flatten_list(li)
    ui = [[1.0*scale]*((len(projected_weights)-nctrl_notchange*2)*2),[1.0]*(len(projected_weights)-nctrl_notchange*2), 0.00,0.00, deltaVel]
    ui = flatten_list(ui)
    yaml_utils.write_parameters_onFile(li=li, ui=ui)
    
    
    """ run new planner, but update the curve only in the next interation        """  
    yaml_utils.write_parameters_onFile(d = (len(projected_weights)-nctrl_notchange*2)*3+2+n_other_opm_var)      
    yaml_utils.write_parameters_onFile(ctrl_points=projected_ctrl_points.tolist(), weights=projected_weights.tolist(), knotvector=new_knotvector_after_split.tolist())
    new_ctrl_points, new_weights = runplanner(VO=VO_flag, verbose=DEBUG)
    new_knotvector = new_knotvector_after_split.copy()
    new_vrobot = yaml_utils.read_yaml()["vcurve"]
    
    

    # nctrl_notchange = 3
    # yaml_utils.write_parameters_onFile(nctrl_notchange=nctrl_notchange)
    
    for _ in range(10):
        # new_ctrl_points, new_weights = runplanner(VO=VO_flag, verbose=DEBUG)
        _cost_constraint = yaml_utils.read_yaml()["_cost_constraint"]
        if _cost_constraint > 0:
            print(f'failed, g = {_cost_constraint:.3f}')
            for iid, obs  in enumerate(list_detected_obstacles):
                    # pos_obs.append([obs.x[-1] + PLANNING_TIME*obs.vx,obs.y[-1]+PLANNING_TIME*obs.vy,obs.r])
                    print(f'obs{iid}, dist = {np.linalg.norm(np.array([obs.x[-1], obs.y[-1]]) - agent_pos[:2]):.4f}')
            new_ctrl_points, new_weights = runplanner(VO=VO_flag, verbose=DEBUG)
            new_vrobot = yaml_utils.read_yaml()["vcurve"]
        else:
            break
    
    

    t_run.append(time.time() - tt)  
    t_list.append(t_sim)
    ctrl_points_list.append(ctrl_points)
    weights_list.append(weights)
    knot_list.append(knot)
    curve_list.append(curve)
    
    

    
    t_sim += PLANNING_TIME
    
   
""" record data """
if SAVE_EXPERIMENT:    
    now = datetime.now()
    current_time = now.strftime("_%Y%m%d_%H%M")        
    path2save = PATH_RESULTS+'experiment_'+str(NAGENTS) + current_time + '.pkl'
    parameters = ["t_list", "t_run", "ctrl_points", "weights", "knot", "curve"
    ]
    dict_parameters = dict(zip(parameters, range(len(parameters))))
    with open(path2save, 'wb') as f: 
        pickle.dump([t_list, t_run, ctrl_points_list, weights_list, knot_list, curve_list, \
                    dict_parameters], f)     

    
print("")
print(rf'avg computational effort = {np.mean(np.array(t_run))} $\pm$ {np.std(np.array(t_run))}s')
print( f'min computational effort = {np.min(np.array(t_run))}s')
print( f'max computational effort = {np.max(np.array(t_run))}s')
fig = plt.figure()
plt.plot(t_run, '*-b')
plt.show()


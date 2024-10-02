#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import time
import re
import subprocess
import csv


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

NAME_EXPERIMENT = "1obs"

yaml_agent = YAML_utils(filename=AGENT_YAML_FILE_NAME)
yaml_opt = YAML_utils(filename=OPT_YAML_FILE_NAME)
yaml_planner = YAML_utils(filename=PLANNER_YAML_FILE_NAME)



def runplanner(VO=True, verbose=True):
    """ Run the path planner DE3D-NURBS (a c++ object) """
    if VO:
        running = "./online_path_planning_denurbs/source/main_VO"
    else:
        running = "./online_path_planning_denurbs/source/main"
    if verbose:
        subprocess.run([running, "1", OPT_YAML_FILE_NAME, AGENT_YAML_FILE_NAME])
    else:
        subprocess.run([running, "1", OPT_YAML_FILE_NAME, AGENT_YAML_FILE_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 
   
    # # new_ctrl_points, weights = yaml_utils.getControlWeightsfromConfig(yaml_utils.read_yaml())
    # new_ctrl_points = np.array(yaml_agent.read_yaml()["control_points"], dtype=float)
    # weights = np.array(yaml_agent.read_yaml()["weights"], dtype=float)

    # return new_ctrl_points, weights




""" Parameters of the simulation and optimization problem """

"""simulation parameters """
tplanner = 0.4

DEBUG = False
VO_flag = True


""" control parameters """
tcontrol = 0.05
Kf_field = 6 #1.95
s_i = 0
s_f = 0.4
s_delta = 0.05

""" robot parameters"""
vrobot = 0.3
# vobs = 0.3
deltaVel = 0.1

# vr = vrobot + deltaVel
# r_control = 1/Kf_field*np.tan(np.pi/2*deltaVel/vr)

r_control = 0.121
r = 0.0
pmin = 0.25


Rsafe = 0.15 + r_control + 0.075
Rview = 1.55


""" optimization parameters """
tVO = 1.2*tplanner
degree = 7
num_samples = 440
n_other_opm_var = 1
nctrl_notchange = 3
scale = 2


""" Create the agent """
robot1 = robot_3dholonomic_velocity(r=r, x0=-2.5, y0=-0.5, z0=0., dt = tcontrol, color='blue', text_id='') 


""" OBSTACLES"""
def update_obstacles(list_static_obs, list_static_vobs, list_dynamic_obs,  list_dynamic_vobs, Rview):
    list_detected_obstacles = list_static_obs.copy()
    list_detected_vobs = list_static_vobs.copy()
    for obs, vobs in zip(list_dynamic_obs, list_dynamic_vobs):
        if np.linalg.norm(robot1.pos - obs.pos) < Rview:
            list_detected_obstacles.append(obs)
            list_detected_vobs.append(vobs)
    return list_detected_obstacles, list_detected_vobs

""" Static obstacles """
N_static_obs = 2
vobs_static = np.array([0,0,0])
pos_static = [
    [.500, .00, 0.00, 0.30],
    # [0.00, 0.20, 0.00, 0.50],
    [-1.3, 0.0, 0.0, 0.5+r_control+0.075],
    # [0.75, 0.75, 0.00, 0.50],
    
    
    
    ]
list_static_obs = []
list_static_vobs = []
for i in range(N_static_obs):
    list_static_obs.append(robot_3dholonomic_velocity(x0=pos_static[i][0], y0=pos_static[i][1], z0=pos_static[i][2], r=pos_static[i][3],  dt = tcontrol, color='gray', text_id=fr'$obs_{i}$'))
    list_static_vobs.append(vobs_static)

""" Dynamic obstacles """
N_dynamic_obs = 0
vobs_dynamic = [
    # np.array([-1,  0,  0])/np.linalg.norm(np.array([-1,  0,  0]))*vobs/4, 
    # np.array([ 0, -1,  0])/np.linalg.norm(np.array([ 0, -1,  0]))*vobs, 
    # np.array([ 1, -1,  0])/np.linalg.norm(np.array([ 1, -1,
    #   0]))*vobs,            
    # np.array([ -1, -1,  0])/np.linalg.norm(np.array([ -1, -1,  0]))*vobs,    
                
                

                
                
                ]
pos_dynamic = [
    # [r + Rview  + 2*(vobs+vrobot)*tplanner, 0.08, 0.00, Rsafe],
    # # [0.75, 1.2, 0.00, Rsafe],
    # [0.75, 1.8, 0.00, Rsafe],
    # [-0.5, 1.5, 0.00, Rsafe],
    # [1.5, 1.5, 0.00, Rsafe],
    
    
    
    
    
    ]
list_dynamic_obs = []
list_dynamic_vobs = []
for i in range(N_dynamic_obs):
    list_dynamic_obs.append(robot_3dholonomic_velocity(x0=pos_dynamic[i][0], y0=pos_dynamic[i][1], z0=pos_dynamic[i][2], r=pos_dynamic[i][3],  dt = tcontrol, color='green', text_id=fr'$obs_{N_dynamic_obs+i-1}$'))
    list_dynamic_vobs.append(vobs_dynamic[i])


""" Setting for the curve """
kappa_max = 1/pmin
gammai = 0*np.pi/180
gammaf = 0*np.pi/180

""" waypoint """
thi = thf = 0
pti = [robot1.x[0], robot1.y[0], 0]
ptf = [1.4, 0.5, 0] # goal position
goal_point = np.array(ptf[:2])

yaml_planner.write_parameters_onFile(target_position=ptf, agent_pos=pti)
yaml_agent.write_parameters_onFile(projected_point=pti)
vi = vf = 0.1
initial_ctrlpoints, _, _, _= pathNURBS.generate_line_points(pti=pti, ptf=ptf, gammai=gammai, gammaf=gammaf, thi=0, thf=0, num_points=4, vi=vi, vf=vf, dimension=2)

wind = [0, 0]
ctrl_points = initial_ctrlpoints.copy()
weights = np.ones(len(ctrl_points))



_, _, curve = pathNURBS.nurbs(degree=degree, points=ctrl_points, weigths=weights, dt=1/num_samples)
initial_curve = curve
knot = curve.knotvector
yaml_planner.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights= weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)
yaml_opt.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights= weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)
yaml_agent.write_parameters_onFile(ctrl_points=ctrl_points.tolist(), weights= weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)

scale = 2        
li = [[-1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[-1.00001]*(len(weights)-nctrl_notchange*2),-1,-1, -deltaVel]
li = flatten_list(li)
ui = [[1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[1.0]*(len(weights)-nctrl_notchange*2), 0.00,0.00, deltaVel]
ui = flatten_list(ui)
yaml_opt.write_parameters_onFile(li=li, ui=ui)



""" Running simulation """
""" List of detected obstacles """
list_detected_obstacles, list_detected_vobs = update_obstacles(list_static_obs, list_static_vobs, list_dynamic_obs,  list_dynamic_vobs, Rview=Rview)
obstacles = []
vel_obs = []
for obs in list_detected_obstacles:
    """ projection of the obstacles"""
    obstacles.append([obs.x[-1]+obs.vx*tplanner,obs.y[-1]+obs.vy*tplanner, obs.z[-1]+obs.vz*tplanner, obs.r])
    vel_obs.append([obs.vx, obs.vy, obs.vz])
obstacles = np.array(obstacles)
vel_obs = np.array(vel_obs)
yaml_opt.write_parameters_onFile(obs=obstacles.tolist(), vobs=vel_obs.tolist())
yaml_planner.write_parameters_onFile(obs=obstacles.tolist(), vobs=vel_obs.tolist())

tsim = 0
frames = []
show_curvature = False
if show_curvature:
    fig = plt.figure(figsize=(14.08,6.08), dpi=100)
else:
    fig = plt.figure(figsize=(6.08,6.08), dpi=100)

viewField = False

""" initial path """
runplanner(VO=VO_flag, verbose=DEBUG)
new_ctrl_points = np.array(yaml_agent.read_yaml()["control_points"], dtype=float)
new_weights = np.array(yaml_agent.read_yaml()["weights"], dtype=float)
new_vrobot = yaml_agent.read_yaml()["vcurve"]

yaml_planner.write_parameters_onFile(ctrl_points=new_ctrl_points.tolist(), weights= new_weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)
yaml_opt.write_parameters_onFile(ctrl_points=new_ctrl_points.tolist(), weights= new_weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)
yaml_agent.write_parameters_onFile(ctrl_points=new_ctrl_points.tolist(), weights= new_weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)
yaml_planner.write_parameters_onFile(start=0)

num_samples = 800
cx,cy, curve = pathNURBS.nurbs(degree=degree, points=new_ctrl_points, weigths=new_weights, dt=1/num_samples)
initial_curve = curve
knot = curve.knotvector

data = np.vstack((cx, cy)).T
print(data)
data = data.tolist()
df = pd.DataFrame(data, columns=['x', 'y'])
df.to_csv("experiment_initial_"+NAME_EXPERIMENT+".csv", index=False)

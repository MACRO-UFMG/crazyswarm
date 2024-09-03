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


INPUT_YAML_FILE_NAME = "config/config.yaml"
yaml_utils = YAML_utils(filename=INPUT_YAML_FILE_NAME)

TAKEOFF_DURATION    = 2
SIMULATION_TIME     = 60
SAMPLING_TIME       = 0.1

""" robot parameters"""
vrobot = 0.4
vobs = 0.3
deltaVel = 0.1

vr = vrobot + deltaVel
r_control = 1/Kf_field*np.tan(np.pi/2*deltaVel/vr)


r = 0.0
pmin = 0.1


Rsafe = 0.2 + r_control 
Rview = 1.2


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    data = []

    # %% Vector Field
    # Set Target Curve

    # %% Build the Vector Field
    ########################
    """ Setting for the curve """
    kappa_max = 1/pmin
    gammai = 45*np.pi/180
    gammaf = 45*np.pi/180

    """ waypoint """
    thi = thf = 0
    pti = [robot1.x[0], robot1.y[0], 0]
    ptf = [1.50, 1.5, 0] # goal position
    goal_point = np.array(ptf[:2])
    vi = vf = 0.1
    initial_ctrlpoints, _, _, _= pathNURBS.generate_line_points(pti=pti, ptf=ptf, gammai=gammai, gammaf=gammaf, thi=0, thf=0, num_points=4, vi=vi, vf=vf, dimension=2)

    wind = [0, 0]
    ctrl_points = initial_ctrlpoints.copy()
    weights = np.ones(len(ctrl_points))

    _, _, curve = pathNURBS.nurbs(degree=degree, points=ctrl_points, weigths=weights, dt=1/num_samples)
    initial_curve = curve
    knot = curve.knotvector
    yaml_utils.write_parameters_onFile(ctrl_points=initial_ctrlpoints.tolist(), weights= weights.tolist(), knotvector=np.array(curve.knotvector).tolist(), num_samples=num_samples, dimension=(len(ctrl_points)-6)*3+2+n_other_opm_var, wind=wind, vuav=vrobot, kappa_max=kappa_max, r_robot=r, vcurve=vrobot)

    scale = 2        
    li = [[-1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[-1.00001]*(len(weights)-nctrl_notchange*2),-1,-1, -deltaVel]
    li = flatten_list(li)
    ui = [[1.0*scale]*((len(weights)-nctrl_notchange*2)*2),[1.0]*(len(weights)-nctrl_notchange*2), 0.00,0.00, deltaVel]
    ui = flatten_list(ui)
    yaml_utils.write_parameters_onFile(li=li, ui=ui)

    """ Update robot using the vector field control """
    curve.delta = 1/15000
    eval_func = curve.evaluate_single
    field, s_star = compute_field(eval_func, pos=robot1.pos[:-1], s_i = s_i, s_f = s_f, Kf=Kf_field, vr=vrobot)
    curve.delta = 1/num_samples
    """ send control (vel) to the robot """
    robot1.step(vx=field[0], vy=field[1], vz=0)
    ########################

    # TAKEOFF
    cf.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    init_time = timeHelper.time()
    
    while timeHelper.time() - init_time < SIMULATION_TIME:
        loop_start = timeHelper.time()

        # Compute Reference
        ###################

        ###################
        # Apply Command
        ###################

        ###################
        
        loop_end = timeHelper.time()
        spent_time = loop_end - loop_start
        timeHelper.sleep(SAMPLING_TIME - spent_time)


    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 't', 'id'])
    df.to_csv("experiment.csv", index=False)

    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)
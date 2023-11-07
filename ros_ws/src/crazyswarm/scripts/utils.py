from RobotModels import *

import numpy as np

def computeReference(position, field, h, t, Ts):
    reference = []
    reference_position = position.copy()
    reference_velocity = np.array([0., 0., 0.])
    
    for _ in range(h):
        reference_position += reference_velocity * Ts
        reference_velocity = field.compute(reference_position, t)
        reference.append(np.concatenate((reference_position, reference_velocity)))
        
        t += Ts
        
    return reference

def build_agents(N_AGENTS, RADIUS, SAMPLING_TIME, CUBE_SIDE):
    agents = []
    while len(agents) < N_AGENTS:
        p = np.random.uniform(-CUBE_SIDE/2, CUBE_SIDE/2, 3)
        agent = Agent(id=len(agents), p=p, v=np.array([0., 0., 0.]), a=np.array([0., 0., 0.]), r=RADIUS, Ts=SAMPLING_TIME)
        
        if agent.collisionCheck(colliders=agents):
            continue

        agents.append(agent)
    return agents

def normalized(x):
    l = np.dot(x, x.T)
    # assert l > 0, (x, l)
    if l == 0: return x
    return x / np.sqrt(l)

def dist_sq(a, b):
    return np.dot(b - a,(b - a).T)
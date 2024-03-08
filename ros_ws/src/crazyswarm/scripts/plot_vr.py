import numpy as np
import pandas as pd
from json import load
from tqdm import tqdm
import plotly.express as px

# %% Load Simulation Parameters from JSON
with open("config2.json", "r", encoding="utf-8") as config_file:
    config = load(config_file)

    experiment_config = config["experiment"]
    agent_config = config["agent"]
    vector_field_config = config["vector_field"]

    # Simulation Aspects
    SAMPLING_TIME = experiment_config["SAMPLING_TIME"]
    SIMULATION_TIME = experiment_config["SIMULATION_TIME"]
    CUBE_SIDE = experiment_config["CUBE_SIDE"]
    # Agent Model Properties
    RADIUS = agent_config["RADIUS"]
    # Vector Field Parameters
    arguments = vector_field_config['arguments']
    curves = vector_field_config["curves"]

df = pd.read_csv('experiment.csv')
df_vel = pd.DataFrame()

agents = df['curve'].unique()
time_stamps = df['t'].unique()

for i in tqdm(range(1, len(time_stamps))):
    t0 = time_stamps[i-1]
    t = time_stamps[i]

    Ts = t - t0

    df_t0 = df.loc[df['t'] == t0] 
    df_t = df.loc[df['t'] == t]

    for agent in agents:
        p_t0 = df_t0.loc[df_t0['curve'] == agent]
        p_t = df_t.loc[df_t['curve'] == agent]
        # print(p_t0)
        # print(p_t)
        # print()

        # print(p_t['x'].iloc[0], p_t0['x'].iloc[0])
        vx = (p_t['x'].iloc[0] - p_t0['x'].iloc[0])/Ts
        vy = (p_t['y'].iloc[0] - p_t0['y'].iloc[0])/Ts
        vz = (p_t['z'].iloc[0] - p_t0['z'].iloc[0])/Ts

        vx_est = (p_t['x'].iloc[0] - p_t0['x'].iloc[0])/SAMPLING_TIME
        vy_est = (p_t['y'].iloc[0] - p_t0['y'].iloc[0])/SAMPLING_TIME
        vz_est = (p_t['z'].iloc[0] - p_t0['z'].iloc[0])/SAMPLING_TIME

        # print(np.array([vx, vy, vz]))
        vr = np.linalg.norm(np.array([vx, vy, vz]))
        vr_est = np.linalg.norm(np.array([vx_est, vy_est, vz_est]))

        row = pd.DataFrame([[vr, t, agent]], columns=['vr', 't', 'agent'])
        agent_est = str(agent) + "_est"
        row_est = pd.DataFrame([[vr_est, t, agent_est]], columns=['vr', 't', 'agent'])

        df_vel = pd.concat([df_vel, row], ignore_index=True)
        df_vel = pd.concat([df_vel, row_est], ignore_index=True)

print(df_vel)

fig = px.line(df_vel,
                x='t',
                y='vr',
                color='agent')
fig.show()
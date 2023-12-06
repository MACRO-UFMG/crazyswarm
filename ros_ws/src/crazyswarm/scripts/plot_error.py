import numpy as np
import pandas as pd
from json import load
from tqdm import tqdm
import plotly.express as px

# %% Load Simulation Parameters from JSON
with open("mpc_ca_demo.json", "r", encoding="utf-8") as config_file:
    config = load(config_file)

    experiment_config = config["experiment"]
    agent_config = config["agent"]
    vector_field_config = config["vector_field"]

    # Simulation Aspects
    SAMPLING_TIME = experiment_config["SAMPLING_TIME"]
    SIMULATION_TIME = experiment_config["SIMULATION_TIME"]
    CUBE_SIDE = experiment_config["CUBE_SIDE"]
    ID_LIST = config["agent"]["ID_LIST"]
    # Agent Model Properties
    RADIUS = agent_config["RADIUS"]
    # Vector Field Parameters
    arguments = vector_field_config['arguments']
    curves = vector_field_config["curves"]

df = pd.read_csv('experiment.csv')
df_error = pd.DataFrame()

parametric_curves = []
for curve in curves:
    parametric_curves.append(eval(f"lambda {', '.join(arguments)}: {curve}"))

s = np.linspace(0, 2*np.pi, 1000)

agents = df['curve'].unique()
time_stamps = df['t'].unique()

for i in tqdm(range(1, len(time_stamps))):
    t = time_stamps[i]

    df_t = df.loc[df['t'] == t]

    for agent in agents:
        p_t = df_t.loc[df_t['curve'] == agent]
        p = np.array([p_t['x'], p_t['y'], p_t['z']])

        curve = parametric_curves[agent](s,t)
        
        distances = np.linalg.norm(curve - p.T, axis=1)
        path_error = np.min(distances)

        row = pd.DataFrame([[path_error, t, agent]], columns=['error', 't', 'agent'])
        df_error = pd.concat([df_error, row], ignore_index=True)

fig = px.line(df_error,
                x='t',
                y='error',
                color='agent')
fig.show()
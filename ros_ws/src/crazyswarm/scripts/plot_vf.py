import numpy as np
import pandas as pd
from json import load
from tqdm import tqdm
import plotly.express as px

# %% Load Simulation Parameters from JSON
with open("config.json", "r", encoding="utf-8") as config_file:
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

parametric_curves = []
for curve in curves:
    parametric_curves.append(eval(f"lambda {', '.join(arguments)}: {curve}"))

s = np.linspace(0, 2*np.pi, 1000)
time_stamps = df['t'].unique()

for t in tqdm(time_stamps):

    for i in range(len(parametric_curves)):
        curve_data = parametric_curves[i](s, t)
        curve_df = pd.DataFrame(curve_data, columns=['x', 'y', 'z'])
        curve_df['t'] = t
        curve_df['curve'] = "curve " + str(i)
        curve_df['mode'] = 0.025
        df = pd.concat([df, curve_df], ignore_index=True)

fig = px.scatter_3d(df,
                x='x',
                y='y',
                z='z',
                range_x=[-CUBE_SIDE/2, CUBE_SIDE/2],
                range_y=[-CUBE_SIDE/2, CUBE_SIDE/2],
                range_z=[-CUBE_SIDE/2, CUBE_SIDE/2],
                color='curve',
                animation_frame='t',
                size='mode', opacity=1)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.update_layout(scene_aspectmode='cube')
fig.show()
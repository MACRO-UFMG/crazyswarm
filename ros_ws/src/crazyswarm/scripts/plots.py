import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from online_path_planning_denurbs.scripts.__utils import *

NAME_EXPERIMENT = "1obs"

pobs0 = np.array([-1.3, 0.0, 0.0, 0.5])
vobs0 = np.array([0.0, 0.0, 0.0])

pobs1 = np.array([.5, .0, 0.0, 0.3])
vdir1 = np.array([-1.5, -0., 0.0]) - pobs1[:3]
vobs1 = 0.1*vdir1/np.linalg.norm(vdir1)

fig = plt.figure()
ax = fig.add_subplot(111)
# Carregar o arquivo CSV
df = pd.read_csv("experiment_"+NAME_EXPERIMENT+".csv")
df_i = pd.read_csv("experiment_initial_"+NAME_EXPERIMENT+".csv")
print(df_i["y"])
# Criar o gráfico de trajetória
# plt.figure(figsize=(8, 6))

# Se houver múltiplos objetos com diferentes 'id', plota a trajetória de cada um
# for obj_id in df['id'].unique():
#     # Filtra os dados para cada id único
#     obj_data = df[df['id'] == obj_id]
#     x = np.array(obj_data['x'],dtype=float)
#     y = np.array(obj_data['y'],dtype=float)
#     # Plota o caminho (x, y)
#     t = np.array(obj_data['t'], dtype=float)
#     ax.plot(obj_data['x'], obj_data['y'], label=f'ID {obj_id}')
#     for ii, ti in enumerate(t):
#         if ii%5 == 0:
#             ax.text(x[ii], y[ii],f'{ti*10:.0f}', fontsize=6)
ax.plot(np.array(df_i["x"]), np.array(df_i["y"]), '--m')
draw_disc(pobs1[:2],pobs1[3], ax=ax, color='g')
draw_disc(pobs0[:2],pobs0[3], ax=ax, color='gray')

# Configurações do gráfico
ax.set_title('Trajetória dos objetos')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.grid(True)
ax.axis('equal')
# Mostrar o gráfico
plt.show()

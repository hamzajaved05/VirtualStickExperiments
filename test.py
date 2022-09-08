from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from util import *
from Plotter import plot_3d
from filterpy.common import Q_discrete_white_noise

clean_pos = getHelicalPath()
filtered = []
f = KalmanFilter (dim_x=9, dim_z=3)
deltaT = 2


pos_noise_factors = np.array([[0.8, 0.8, 1]])
measurement_pos = add_measurement_noise(clean_pos, pos_noise_factors)

vel_clean = (clean_pos[1:] - clean_pos[:-1]) / deltaT 
vel_noise_factors = np.array([[0.4, 0.4, 0.1]])
measurement_vel = add_measurement_noise(vel_clean, vel_noise_factors)


measurement_vector = np.zeros(9)
measurement_vector[:3] = measurement_pos[0]
f.x = measurement_vector
f.F = getStateSVA(deltaT)
f.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]])
f.P *= 0.01
f.R = np.eye(3) * 0.5




filtered.append(f.x)
for measurement in measurement_pos[1:]:
    measurement_vector = np.zeros(9)
    measurement_vector[:3] = measurement
    f.predict()
    f.update(measurement)
    filtered.append(f.x)

filtered = np.array(filtered)

filtered_error = np.mean(np.linalg.norm(clean_pos - filtered[:, :3], axis = 1))
measurement_error = np.mean(np.linalg.norm(clean_pos - measurement_pos, axis = 1))

print(f"Filtered error : {filtered_error}, measurement_error: {measurement_error}")

plot_3d(clean_pos, newfigure = True, suppress = True, plotfn = "line", marker = "*")
plot_3d(measurement_pos, newfigure = False, suppress = True, plotfn = "line", marker = ".")
plot_3d(filtered, newfigure = False, suppress = False, plotfn = "line", marker = "v")

print("Finished")
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from util3d import *
from Plotter import plot_3d, plot_2d
from filterpy.common import Q_discrete_white_noise

clean_pos = getHelicalPath()
filtered = []
KalFil = KalmanFilter(dim_x=9, dim_z=6, dim_u=3)
deltaT = 3

pos_noise_factors = np.array([[1, 1, 0.5]])
measurement_pos = add_measurement_noise(clean_pos, pos_noise_factors)[2:]

vel_clean = (clean_pos[1:] - clean_pos[:-1]) / deltaT
vel_noise_factors = np.array([[8e-5, 8e-5, 0.00005]])
measurement_vel = add_measurement_noise(vel_clean, vel_noise_factors)[1:]
action_vel = add_measurement_noise(vel_clean, vel_noise_factors)[1:]

acc_clean = (vel_clean[1:] - vel_clean[:-1]) / 2
acc_noise_factors = np.array([[5e-8, 5e-8, 1e-18]])
measurement_acc = add_measurement_noise(acc_clean, acc_noise_factors)

KalFil.F = getStateSVA(deltaT)
x_init = create_initstate_vector(measurement_pos[0], measurement_vel[0], measurement_acc[0])
KalFil.x = x_init
KalFil.B = createControlMatrix(deltaT)
KalFil.H = createMeasurementMatrix(deltaT)
KalFil.P *= 0.01
KalFil.R = np.eye(6) * 0.1
KalFil.Q = np.eye(9) * 0.001
KalFil.Q[:3, :3] *= 0.1
KalFil.Q[3:6, 3:6] *= (0.5 / deltaT)
KalFil.Q[6:9, 6:9] *= (0.5 / deltaT**2)


filtered = []
predictions = []
gains = []
residuals = []
for it, (s, v, a, u) in enumerate(zip(measurement_pos, measurement_vel, measurement_acc, action_vel)):

    KalFil.predict(u=create_control_vector(s, v, a, u))
    predictions.append(KalFil.x_prior)

    KalFil.update(z=create_measurement_vector(s, v, a, u))
    filtered.append(KalFil.x)
    gains.append(KalFil.K)
    residuals.append(KalFil.y)

filtered = np.array(filtered)
gains = np.array(gains)
residuals = np.array(residuals)
predictions = np.array(predictions)

filtered_error = np.mean(np.linalg.norm(clean_pos[2:, :2] - filtered[:, :2], axis=1))
measurement_error = np.mean(np.linalg.norm(clean_pos[2:, :2] - measurement_pos[:, :2], axis=1))

print(f"Filtered error : {filtered_error}, measurement_error: {measurement_error}")

plot_2d(predictions, newfigure=True, suppress=True, plotfn="line", marker=".")
plot_2d(measurement_pos, newfigure=False, suppress=True, plotfn="line", marker="v")
plot_2d(filtered, newfigure=False, suppress=False, plotfn="line", marker="*")

print("Finished")

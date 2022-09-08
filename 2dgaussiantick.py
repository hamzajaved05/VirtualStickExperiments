import matplotlib.pyplot as plt
import numpy as np
from Plotter import plot_3d, plot_2d


def getgauss(mean, cov, pos):
    return 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov))) *  np.exp((-1 / 2) * (pos - mean).dot(np.linalg.inv(cov)).dot((pos - mean).T))

def getgaussnorm(mean, cov, pos):
    return np.exp((-1 / 2) * ((pos - mean).dot(np.linalg.inv(cov)) * (pos - mean)).sum(-1))

mean = np.array([5, 5])
cov = np.array([[0.1, 0], [0, 2]])

values = np.linspace(2, 8, 50)

x, y = np.meshgrid(values, values)
points = np.stack([x, y]).transpose(1, 2, 0).reshape(-1, 2)
getgauss(mean, cov, points[0])
z = getgaussnorm(mean, cov, points).reshape(-1, 1)
# z =  np.array([getgauss(mean, cov,pt ) for pt in points]).reshape(-1, 1)
points = np.concatenate([points, z], axis = -1)
plot_3d(points, plot_args={"c": z, "s": 0.4})

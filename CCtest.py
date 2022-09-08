import numpy as np
from CarrotChasing import CarrotChasing
from Drone import DroneProf
from VelocityProfile import VelocityProfile
from util3d import getHelicalPath, getLinearPath, getangle, getStraightPath, getZigZag
from Plotter import plot_2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from src import WayPointer
plt.ion()

name = "imgs/00exp__Log_circle"
data: np.ndarray = getHelicalPath()[:, :2]
locking = True
ret, (fig, ax) = plot_2d(data, plotfn= "line", suppress = True)
plt.show(block = False)
Drone = DroneProf()
WP = WayPointer(data, locking = True)
Drone.setPos(WP.getTargetWP())
prev = Drone.pos
velocityprofiler = VelocityProfile(distance = 1, c= 0.5, v_max = 15, v_min=0.5)
# velocityprofiler(np.pi / 2, 10)
CC = CarrotChasing(vel = velocityprofiler, delta = 1, locking = locking)
prevWP = 0
nextWP = 1
timeDelta = 0.15
poses, velocities, accs = [], [], []
cmap = cm.get_cmap("brg")
iteration = 0

while True:
    angle = getangle(data[prevWP], data[nextWP], data[nextWP + 1])
    angle = getangle(WP.getCurrentWP(), WP.getTargetWP(), WP.getnexttoTargetWP())

    vel, orientation, VTP = CC.solve(prev, data[prevWP], data[nextWP], angle)
    vel, orientation, VTP = CC.solve(Drone.pos, WP.getCurrentWP(), WP.getTargetWP(), angle)

    vel_oriented = vel * np.asarray([np.cos(orientation), np.sin(orientation)])
    Drone.targetvel(vel_oriented, timeDelta)
    prev = Drone.pos
    # if random.random() < 0.05:
    #     prev += np.random.rand(2)
    poses.append(Drone.pos)

    if len(poses) > 1:
        velocities.append((poses[-1] - poses[-2]) / timeDelta)

    if len(velocities) > 1:
        accs.append((velocities[-1] - velocities [-2]) / timeDelta)

    iteration += 1

    if WP.RegistrationCheck(Drone.pos):
        ax.annotate(f"{iteration * timeDelta:.1f}", [prev[0],prev[1]])
        if WP.getTargetWP() is None:
            break

    # if locking and (np.linalg.norm(data[nextWP] - prev) < 0.25) or (not locking and np.linalg.norm(data[nextWP + 1] - prev) < np.linalg.norm(data[nextWP] - data[nextWP + 1])):
    #     prevWP += 1
    #     nextWP += 1
    #     ax.annotate(f"{iteration * timeDelta:.1f}", [prev[0],prev[1]])
    #     if nextWP == len(data) - 1:
    #         break

    plot_2d(np.asarray(poses[-2:]), newfigure= False, plotfn = "line", plot_args = {"c": cmap(vel / velocityprofiler.v_max)})
    plt.draw()
    plt.pause(0.0001)

plt.savefig(f"{name}_1.jpeg")
plt.clf()
plt.plot(np.linspace(0, iteration * timeDelta, len(velocities)), np.linalg.norm(velocities, axis = 1))
plt.savefig(f"{name}_2.jpeg")
plt.clf()
plt.plot(np.linspace(0, iteration * timeDelta, len(accs)), np.linalg.norm(accs, axis = 1))
plt.savefig(f"{name}_3.jpeg")
plt.clf()
plt.plot(np.linspace(0, iteration * timeDelta, len(accs)), np.linalg.norm(accs, axis = 1) * np.sign(np.mean(accs, axis = 1)))
plt.savefig(f"{name}_4.jpeg")
print("Completed")

import numpy as np


class DroneProf:
    def __init__(self, dim = 2):
        self.dim = 2
        self.pos = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        self.rate = 3

    def setPos(self, pos):
        self.pos = pos

    def setvel(self, vel):
        self.vel = vel

    def setAcc(self, acc):
        self.acc = acc

    def targetvel(self, vel, dT):
        self.acc = (vel - self.vel) * (self.rate / np.linalg.norm(vel - self.vel))
        self.pos = self.pos + self.vel * dT + 0.5 * self.acc * dT**2
        self.vel = self.vel + self.acc * dT

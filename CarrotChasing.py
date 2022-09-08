import numpy as np
import VelocityProfile

class CarrotChasing():
    def __init__(self, vel: VelocityProfile, K: float = 0.5, delta: float = 1.0, locking: bool = False):
        self.K: float = K
        self.delta: float = delta
        self.currentPos = None
        self.ClosestWaypoint1 = None
        self.ClosestWaypoint2 = None
        self.VTP = None
        self.locking = locking
        self.vel = vel

    def solve(self, currentPos: np.ndarray, ClosestWaypoint1: np.ndarray, ClosestWaypoint2: np.ndarray, angle: float):
        self.currentPos = currentPos
        self.ClosestWaypoint1 = ClosestWaypoint1
        self.ClosestWaypoint2 = ClosestWaypoint2

        self.VTP, distanceTo, distanceFrom = self.getS()
        self.corrYaw = self.getCorrectionYaw()
        AbsdistanceTo = self.getL2error2WP2()
        vel = self.vel(angle, min(distanceTo, AbsdistanceTo))
        # vel = 5
        self.vel_value = vel * (1)
        return self.vel_value, self.corrYaw, self.VTP

    def getL2error(self,):
        return np.linalg.norm(self.currentPos - self.ClosestWaypoint1)

    def getL2error2WP2(self,):
        return np.linalg.norm(self.currentPos - self.ClosestWaypoint2)

    def getL2Waypoints(self,):
        return np.linalg.norm(self.ClosestWaypoint2 - self.ClosestWaypoint1)

    def getTheta(self):
        difference = self.ClosestWaypoint2 - self.ClosestWaypoint1
        return np.arctan2(difference[1], difference[0]) % (2*  np.pi)

    def getThetaU(self):
        difference = self.currentPos - self.ClosestWaypoint1
        return np.arctan2(difference[1], difference[0])  % (2 * np.pi)

    def getBeta(self):
        return self.getTheta() - self.getThetaU()

    def getRandE(self):
        return np.abs(self.getL2error() * np.cos(self.getBeta())), np.abs(self.getL2error() * np.sin(self.getBeta()))

    def getS(self):
        self.R, self.E = self.getRandE()
        theta = self.getTheta()
        distance = self.getL2Waypoints()

        # Adjust if the target might exceed the WP
        if self.R + self.delta > distance and self.locking and self.E > 1:
            self.R -= distance - self.R

        #Return WP + cartesianOffset
        return self.ClosestWaypoint1 +  np.array([(self.R + self.delta)*np.cos(theta), (self.R + self.delta) * np.sin(theta)]), distance - self.R, self.R

    def getCorrectionYaw(self):
        difference = self.VTP - self.currentPos
        return np.arctan2(difference[1], difference[0]) % (2 * np.pi)

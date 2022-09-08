import numpy as np


class WayPointer(object):

    def __init__(self, waypoints: np.ndarray, locking: bool = True):
        """
        The __init__ function is the constructor for a class. It is called whenever an instance of a class is created.
        The __init__ function can take arguments, but self is always the first one. Self is just a reference to the instance
        of the class (i.e., it refers to "this" object in C++). The __init__ function can take any number of arguments, and
        should return nothing.

        :param self: Used to Refer to the object itself.
        :param waypoints:np.ndarray: Used to Store the waypoints that are passed to the constructor.
        :return: The waypoints.

        :doc-author: Trelent
        """
        self.waypoints: np.ndarray = waypoints
        self.currentIdx: int = -1
        self.targetIdx: int = 0
        self.locking = locking
        self.lastPos: np.ndarray = None

    def getCurrentWP(self,):
        return self.waypoints[self.currentIdx]

    def getTargetWP(self):
        if self.targetIdx < len(self.waypoints):
            return self.waypoints[self.targetIdx]
        else:
            return None

    def getnexttoTargetWP(self):
        if self.targetIdx + 1 < len(self.waypoints):
            return self.waypoints[self.targetIdx + 1]
        else:
            return None

    def distancefromCurrent(self, pos):
        return np.linalg.norm(self.waypoints[self.currentIdx] - pos)

    def distancetoTarget(self, pos):
        return np.linalg.norm(self.waypoints[self.targetIdx] - pos)

    def RegistrationCheck(self, pos):
        if self.locking:
            if (self.distancetoTarget(pos) < 0.4):
                self.currentIdx += 1
                self.targetIdx += 1
                return True
        else:
            if np.linalg.norm(self.waypoints[self.targetIdx + 1] - pos) < np.linalg.norm(self.waypoints[self.targetIdx] - self.waypoints[self.targetIdx + 1]):
                self.currentIdx += 1
                self.targetIdx += 1
                return True
        return False

    def __repr__(self):
        return f"WayPoints Count: {len(self.waypoints)}, Current Waypoint: {self.currentIdx}: {self.waypoints[self.currentIdx]}"

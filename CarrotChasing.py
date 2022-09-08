import numpy as np
import VelocityProfile

class CarrotChasing():
    def __init__(self, vel: VelocityProfile, K: float = 0.5, delta: float = 1.0, locking: bool = False):
        """
        The __init__ function is called when an instance of the class is created.
        It initializes all of the variables in the class and sets them to their default values.

        :param self: Used to Reference the object to which the function is being called.
        :param vel:VelocityProfile: Used to Pass the velocity profile to the controller.
        :param K:float=0.5: Used to Set the gain of the controller.
        :param delta:float=1.0: Used to Define the lookahead distance.
        :param locking:bool=False: Used to Determine whether the vehicle should be locked to its current waypoint or not.
        :return: The __init__ function itself.

        :doc-author: Trelent
        """
        self.K: float = K
        self.delta: float = delta
        self.currentPos = None
        self.ClosestWaypoint1 = None
        self.ClosestWaypoint2 = None
        self.VTP = None
        self.locking = locking
        self.vel = vel

    def solve(self, currentPos: np.ndarray, ClosestWaypoint1: np.ndarray, ClosestWaypoint2: np.ndarray, angle: float):
        """
        The solve function takes in the current position of the car,
        the closest waypoint to that position, and the angle between those two points.
        It then returns a tuple containing:
            - The velocity value for this iteration of control (scalar)
            - The yaw correction value for this iteration of control (scalar)

        :param self: Used to Access variables that belongs to the class.
        :param currentPos:np.ndarray: Used to Store the current position of the car.
        :param ClosestWaypoint1:np.ndarray: Used to Get the closest waypoint to the current position.
        :param ClosestWaypoint2:np.ndarray: Used to Calculate the distance from the car to a waypoint.
        :param angle:float: Used to Calculate the steering angle.
        :return: The velocity and the yaw angle for the next step.

        :doc-author: Trelent
        """
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
        """
        The getL2error function returns the L2 error between the current position and
        the closest waypoint. The L2 error is defined as:
        L2 = sqrt( (x_current - x_waypoint)^T * (x_current - x_waypoint) )

        :param self: Used to Access variables that belongs to the class.
        :param : Used to Calculate the l2 error between the current position and the closest waypoint.
        :return: The l2 error between the current position and the closest waypoint.

        :doc-author: Trelent
        """
        return np.linalg.norm(self.currentPos - self.ClosestWaypoint1)

    def getL2error2WP2(self,):
        """
        The getL2error2WP2 function returns the L2 norm of the vector between the current position and
        the closest waypoint. This is a helpful function for calculating how far away from a waypoint we are.

        :param self: Used to Access variables that belongs to the class.
        :param : Used to Calculate the distance between the current position and the closest waypoint.
        :return: The error between the current position and the closest waypoint to it.

        :doc-author: Trelent
        """
        return np.linalg.norm(self.currentPos - self.ClosestWaypoint2)

    def getL2Waypoints(self,):
        """
        The getL2Waypoints function returns the distance between two waypoints.
        The first waypoint is the closest one to the vehicle, and the second is selected ahead of it along
        the path. The function returns this distance.

        :param self: Used to Access the class variables.
        :param : Used to Calculate the distance between the two closest waypoints.
        :return: The distance between the two closest waypoints in the current lap.

        :doc-author: Trelent
        """
        return np.linalg.norm(self.ClosestWaypoint2 - self.ClosestWaypoint1)

    def getTheta(self):
        """
        The getTheta function returns the angle between the current waypoint and the next waypoint.
        The function takes in two parameters, self.ClosestWaypoint2 and self.ClosestWaypoint 1, which are both integers that represent indices of a list of waypoints (self._waypoints). The function returns an angle in radians.

        :param self: Used to Access the variables defined in the class.
        :return: The angle of the vector between two waypoints.

        :doc-author: Trelent
        """
        difference = self.ClosestWaypoint2 - self.ClosestWaypoint1
        return np.arctan2(difference[1], difference[0]) % (2*  np.pi)

    def getThetaU(self):
        """
        The getThetaU function returns the angle between the current position and
        the closest waypoint. The function returns a value in radians, with 0 being
        straight ahead of the car and positive values meaning that it is turning to
        the left. This function is used by our controller to determine how much it needs
        to turn.

        :param self: Used to Access the variables of the class within a function.
        :return: The angle between the car and the closest waypoint.

        :doc-author: Trelent
        """
        difference = self.currentPos - self.ClosestWaypoint1
        return np.arctan2(difference[1], difference[0])  % (2 * np.pi)

    def getBeta(self):
        """
        The getBeta function returns the beta value of a given node.
        The beta value is calculated by subtracting thetaU from theta.

        :param self: Used to Access the attributes and methods of the class in python.
        :return: The beta value of the parameter.

        :doc-author: Trelent
        """
        return self.getTheta() - self.getThetaU()

    def getRandE(self):
        """
        The getRandE function returns the random error in the x and y directions.
        The random error is calculated by taking the L2error of each direction, multiplying it by cos(beta) or sin(beta), respectively, and then returning that value.

        :param self: Used to Access the class attributes.
        :return: The random error in the x and y directions.

        :doc-author: Trelent
        """
        return np.abs(self.getL2error() * np.cos(self.getBeta())), np.abs(self.getL2error() * np.sin(self.getBeta()))

    def getS(self):
        """
        The getS function returns the S vector, which is a vector that points from the car to the closest waypoint.
        The function also returns distance, which is a scalar value representing how far away from the closest waypoint
        the car currently is. The function also returns R, which represents how far away from (or past)
        the closest waypoint we are when we start our locking algorithm.

        :param self: Used to Access variables that belongs to the class.
        :return: The closest waypoint to the car, plus an offset.

        :doc-author: Trelent
        """
        self.R, self.E = self.getRandE()
        theta = self.getTheta()
        distance = self.getL2Waypoints()

        # Adjust if the target might exceed the WP
        if self.R + self.delta > distance and self.locking and self.E > 1:
            self.R -= distance - self.R

        #Return WP + cartesianOffset
        return self.ClosestWaypoint1 +  np.array([(self.R + self.delta)*np.cos(theta), (self.R + self.delta) * np.sin(theta)]), distance - self.R, self.R

    def getCorrectionYaw(self):
        """
        The getCorrectionYaw function returns the yaw angle needed to correct the current position of the robot.
        The function takes in two arguments: self and currentPos. The function returns a single argument, correctionYaw.

        :param self: Used to Access the variables and methods of the class in python.
        :return: The difference between the current position and the desired position in radians.

        :doc-author: Trelent
        """
        difference = self.VTP - self.currentPos
        return np.arctan2(difference[1], difference[0]) % (2 * np.pi)

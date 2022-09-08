import numpy as np


class DroneProf:
    def __init__(self, dim = 2):
        """
        The __init__ function is called automatically every time the class is instantiated.
        It sets up all of the attributes that will be used later on, such as dim (dimension), pos (position), vel (velocity) and acc(acceleration).
        The rate attribute is a variable that determines how fast the object moves along each axis.

        :param self: Used to Refer to the object itself.
        :param dim=2: Used to Define the dimension of the object.
        :return: Self.

        :doc-author: Trelent
        """
        self.dim = 2
        self.pos = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        self.rate = 2

    def setPos(self, pos):
        """
        The setPos function sets the position of the object to a new value.

        :param self: Used to Refer to the object itself.
        :param pos: Used to Set the position of the object.
        :return: The object itself.

        :doc-author: Trelent
        """
        self.pos = pos

    def setvel(self, vel):
        """
        The setvel function sets the velocity of the motor.

        :param self: Used to Refer to the object itself.
        :param vel: Used to Set the velocity of the object.
        :return: The value of the vel parameter.

        :doc-author: Trelent
        """
        self.vel = vel

    def setAcc(self, acc):
        """
        The setAcc function sets the acceleration of the motor.
           @param acc The desired acceleration in units of mm/s^2.

        :param self: Used to Refer to the object itself.
        :param acc: Used to Set the acceleration of the motor.
        :return: The value of acc.

        :doc-author: Trelent
        """
        self.acc = acc

    def targetvel(self, vel, dT):
        """
        The targetvel function takes in a velocity vector and a time step.
        It then calculates the acceleration needed to reach that velocity in the given time step,
        and uses this acceleration to update both the position and velocity vectors of our object.

        :param self: Used to Access the object's attributes.
        :param vel: Used to Calculate the acceleration.
        :param dT: Used to Calculate the time difference between the current frame and the previous frame.
        :return: The acceleration of the object and updates its position, velocity and acceleration.

        :doc-author: Trelent
        """
        self.acc = (vel - self.vel) * (self.rate / np.linalg.norm(vel - self.vel))
        self.pos = self.pos + self.vel * dT + 0.5 * self.acc * dT**2
        self.vel = self.vel + self.acc * dT

    def __repr__(self):
        return f"Drone Pos: {self.pos}, Vel: {self.vel}, Acceleration: {self.acc}, Max allowed acceleration: {self.rate}"

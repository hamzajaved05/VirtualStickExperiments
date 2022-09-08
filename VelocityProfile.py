import numpy as np

class VelocityProfile():
    def __init__(self, distance : float = 1, v_max: float = 15., c: float = 0.5, v_min = 0.5):
        """
        The __init__ function is called automatically every time the class is instantiated.
        It sets up all of the attributes that will be used by all other functions in this class.

        :param self: Used to Refer to the object itself.
        :param distance:float=1: Used to Define the distance between two cars.
        :param v_max:float=15.: Used to Set the maximum velocity of the cars.
        :param c:float=0.5: Used to Determine the maximum velocity of the car.
        :param v_min=0.5: Used to Set the minimum velocity of the cars.
        :return: Nothing.

        :doc-author: Trelent
        """
        self.distance = distance
        self.v_max = v_max
        self.c = c
        self.v_min_abs = v_min
        self.v_diff = self.v_max - self.v_min_abs
        self.func = self.getLogVelocityforCurveIn

    def getLinearVelocityforCurveIn(self, angle: float, distance: float):
        """
        The getLinearVelocityforCurveIn function returns the linear velocity for a curve in.
        The function takes two arguments, angle and distance. The angle is the current heading of the car,
        and distance is how far away from this curve in we are (in meters). The function returns a float
        representing the linear velocity that should be used to drive along this curve in.

        :param self: Used to Reference the class object itself.
        :param angle:float: Used to Calculate the velocity of the car.
        :param distance:float: Used to Calculate the velocity at a certain distance.
        :return: The velocity for a given angle and distance.

        :doc-author: Trelent
        """
        v_min = self.v_min_abs + self.v_diff * np.max([np.cos(np.pi + angle), 0])
        if distance < self.distance:
            return v_min
        else:
            return np.min([v_min + (distance - self.distance) * self.c, self.v_max])

    def getLogVelocityforCurveIn(self, angle: float, distance: float):
        """
        The getLogVelocityforCurveIn function takes in an angle and a distance.
        It returns the velocity of the car at that point on the curve, assuming it is driving at max speed.

        :param self: Used to Reference the class instance.
        :param angle:float: Used to Calculate the velocity of the car.
        :param distance:float: Used to Calculate the velocity of the car.
        :return: The velocity for a given angle and distance.

        :doc-author: Trelent
        """
        v_min = self.v_min_abs + self.v_diff * np.max([np.cos(np.pi + angle), 0])
        if distance < self.distance:
            return v_min
        elif distance < 500:
            return np.min([2 * np.log((distance + 1)), self.v_max])
        else:
            return np.min([(v_min + (distance - self.distance) * self.c), self.v_max])

    def getConstant(self):
        """
        The getConstant function returns the maximum value of the array.

        :param self: Used to Access variables that belongs to the class.
        :return: The value of the constant v_max.

        :doc-author: Trelent
        """
        return self.v_max

    def __call__(self, *args, **kwargs):
        """
        The __call__ function is a special function that allows an object to be called just like a function.
        It also has the ability to run code when it is called, making it ideal for classes which need to have some
        startup code run when they are created.

        :param self: Used to Access variables that belongs to the object.
        :param *args: Used to Pass a non-keyworded, variable-length argument list.
        :param **kwargs: Used to Pass keyworded variable length of arguments to a function.
        :return: The function object.

        :doc-author: Trelent
        """
        return self.func(*args, **kwargs)

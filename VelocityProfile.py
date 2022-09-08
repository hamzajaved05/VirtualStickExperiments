import numpy as np


class VelocityProfile():
    def __init__(self, distance : float = 1, v_max: float = 15., c: float = 0.5, v_min = 0.5):
        self.distance = distance
        self.v_max = v_max
        self.c = c
        self.v_min_abs = v_min
        self.func = self.getLogVelocityforCurveIn

    def getLinearVelocityforCurveIn(self, angle: float, distance: float):
        v_min = self.v_min_abs + (self.v_max - self.v_min_abs) * np.max([np.cos(np.pi + angle), 0])
        if distance < self.distance:
            return v_min
        else:
            return np.min([v_min + (distance - self.distance) * self.c, self.v_max])

    def getLogVelocityforCurveIn(self, angle: float, distance: float):
        v_min = self.v_min_abs + (self.v_max - self.v_min_abs) * np.max([np.cos(np.pi + angle), 0])
        if distance < self.distance:
            return v_min
        elif distance < 100:
            return np.min([2.5 * np.log((distance + 1)), self.v_max])
        else:
            return np.min([(v_min + (distance - self.distance) * self.c), self.v_max])


    def getConstant(self):
        return self.v_max

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

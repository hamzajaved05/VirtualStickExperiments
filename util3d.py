import numpy as np


def getangle(pt1, pt2, pt3):
    if pt3 is None:
        return 0
    r1 = pt2 - pt1
    r2 = pt3 - pt2
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    return np.arccos(np.dot(r1, r2))

def getHelicalPath():
    elements = 25
    revs = 1
    radius = np.linspace(5, 10, elements)
    height = np.linspace(100, 150, elements)
    height_acc = np.linspace(0, 100, elements)
    imagesperrev = 50
    angle = np.linspace(0, 360 * revs, elements)
    angle %= 360
    angle = angle * np.pi / 180
    x, y, z = [], [], []
    for it, (r, h, a, o) in enumerate(zip(radius, height, angle, height_acc)):
        x.append(r*np.cos(a))
        y.append(r*np.sin(a))
        z.append(h + o * it / elements)

    return np.stack([x, y, z]).T


def getStraightPath():
    elements = 2
    x1 = np.linspace(0, 100, elements)
    y1 = np.linspace(0, 0, elements)
    z1 = np.linspace(0, 0, elements)

    x2 = np.linspace(100, 101, elements)
    y2 = np.linspace(0, 100, elements)
    z2 = np.linspace(0, 0, elements)

    x = np.stack([x1, x2]).reshape(-1)
    y = np.stack([y1, y2]).reshape(-1)
    z = np.stack([z1, z2]).reshape(-1)

    return np.stack([x, y, z]).T

def getZigZag():
    x = np.asarray([0, 5, 10, 100, 30, 8 ])
    y = np.asarray([0, 0, 10, 100, 0, -5])
    z = np.asarray([0, 5, 10, 100, 0, 8])
    return np.stack([x, y, z]).T


def getLinearPath():
    elements = 3000
    pos = np.zeros((elements, 3))
    pos[:1000, 0] = np.linspace(0, 50, 1000)
    pos[1000:2000, 1] = np.linspace(0, 60, 1000)
    pos[1000:2000, 0] = 50
    pos[2000:3000, 0] = np.linspace(50, 100, 1000)
    pos[2000:3000, 1] = (pos[2000:3000, 0] - 50) * 0.2 + 60
    return pos


def getStateSVA(deltaT, dim = 3):
    if dim == 2:
        state = np.eye(6)
        indicesS = np.diag_indices(6)
        indicesV = (indicesS[0][:4], indicesS[1][:4] + 2)
        indicesA = (indicesS[0][:2], indicesS[1][:2] + 4)
        state[indicesV] = deltaT
        # state[indicesA] = deltaT**2 / 2
    else:
        state = np.eye(9)
        indicesS = np.diag_indices(9)
        indicesV = (indicesS[0][:6], indicesS[1][:6] + 3)
        indicesA = (indicesS[0][:3], indicesS[1][:3] + 6)
        state[indicesV] = deltaT
        state[indicesA] = deltaT**2 / 2
    return state

def add_measurement_noise(data: np.array, error_factor: np.array):
    return data + (np.random.rand(*data.shape) * error_factor - error_factor / 2)

def create_initstate_vector(p: np.array, v: np.array, a: np.array):
    return np.hstack([p, v, a])

def createControlMatrix(deltaT, dim = 3):
    if dim == 2:
        B = np.zeros((6, 6))
        B[2:4, 2:4] = np.eye(2) * deltaT
    else:
        B = np.zeros((9, 3))
        B[3:6, :] = np.eye(3) * deltaT
    return B

def createMeasurementMatrix(deltaT, dim = 2):
    if dim == 2:
        H = np.zeros((6, 6))
        H[:2,:2] = np.eye(2)
        # H[4:, 4:] = np.eye(2)
    else:
        H = np.zeros((6, 9))
        H[:3,:3] = np.eye(3)
        H[3:, 6:] = np.eye(3)
    return H

def create_control_vector(s, v, a, u):
    return np.hstack([s, v, a])

def create_measurement_vector(s, v, a, u):
    return np.hstack([s, v, a])

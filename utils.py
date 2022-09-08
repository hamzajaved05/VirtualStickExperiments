import os
import pandas as pd
import numpy as np
from Plotter import plot_2d, plot_3d

def obtainValidVSData(csv):
    csv = filtercolums(csv)
    csv = filterZeroed(csv)
    return csv

def filtercolums(csv):
    return csv.iloc[:, 73:]

def filterZeroed(csv):
    return csv[csv.iloc[:,0] != 0]

def getTime(csv, id = 0):
    return ((csv.iloc[:, id] - csv.iloc[0, id]) / 1000).to_numpy()

def getPositions(csv, ids = [3, 6]):
    data = csv.iloc[:, ids[0]:ids[1]].to_numpy()
    return data - data[0:1, :]

def getVelocties(csv, ids = [15, 18]):
    data = csv.iloc[:, ids[0]:ids[1]].to_numpy()
    return data

def readcsv(path = "data/Launch_Time__11_35_orbits_nortk_3ms.csv"):
    csv = pd.read_csv(path)
    csv = obtainValidVSData(csv)
    T = getTime(csv)
    P = getPositions(csv)
    V = getVelocties(csv)
    dT = T[1:] - T[:-1]
    pred = P[:-1] + dT.reshape(-1, 1) * V[:-1]

    error = pred - P[1:]
    plot_2d(P, plot_args={"c":np.linalg.norm(V, axis = 1)})
    return csv


if __name__ == "__main__":
    readcsv()

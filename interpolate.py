import os
import time
import cv2
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
from metrics import euclidean
from KNN import KnearestNeighbors
from scipy.interpolate import interp1d
from yolo import YOLO

with open('/media/mythra/DATA/Users/Karina/Dataset/Up/20.txt') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    X = np.array(x[0:-1])

    y = [float(line.split()[1]) for line in lines]
    Y = np.array(y[0:-1])

    t = np.arange(0,int(X.shape[0]))
    ty = np.arange(0,int(Y.shape[0]))

    fx = interp1d(t, X, kind="cubic")
    fy = interp1d(t, Y, kind="cubic")

    tnew = np.arange(0,int(X.shape[0]-1), 20)
    xinterpolated = fx(tnew)
    
    tynew = np.arange(0,int(Y.shape[0]-1), 20)
    yinterpolated = fy(tynew)
  
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(111)

    plt.axis('equal')
    ax1.plot(x, y, '-', xinterpolated, yinterpolated, 'o')

    plt.show()



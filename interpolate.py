import os
import time
import cv2
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


with open('/media/mythra/DATA/Users/Karina/Dataset/Down/2.txt') as f:
    lines = f.readlines()       # reading the files
    x = [float(line.split()[0]) for line in lines]
    X = np.array(x[0:-1])

    y = [float(line.split()[1]) for line in lines]
    Y = np.array(y[0:-1])
   
    # normalizing the values of X and Y numpy arrays
    normalizedX = X/max(X)
    normalizedY = Y/max(Y)
    

    # converting data elements of arrays into lists
    var = normalizedX.tolist()
    var2 = normalizedY.tolist()

    # getting values within an interval of a list
    t = np.arange(0, len(var)) 
    ty = np.arange(0, len(var2))
       
    # interpolating the values
    fx = interp1d(t, var, kind="linear")
    fy = interp1d(ty, var2, kind="linear")

    # getting values within an interval of a list and interpolating to 20 frames
    tnew = np.arange(0, len(var), 20)
    xinterpolated = fx(tnew)
    
    tynew = np.arange(0, len(var2), 20)
    yinterpolated = fy(tynew)


    # creting a new figure and plotting the values
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(111)

    plt.axis('equal')
    ax1.plot(x, y, '-', xinterpolated, yinterpolated, 'o')

    plt.show()



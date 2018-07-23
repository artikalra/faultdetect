import util.filereader as rd
#from scipy.integrate import quad
import math
import numpy as np
from data.cluster import Clustering


if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")
# to compute to integration of the square of L2norm for all the trajectories
    from sympy import *
    finaldistances = []
    for m in range(1, 11):
        for n in range(1, 11):
            # dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(n + 1)][n])])
            #print(samples._data[:1][0][4:7]) #to use only for the accelerometer components
            dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:n][n - 1])])
            t = Symbol('t')
            ans = integrate(dist, (t, samples._data[:m][m-1][0], samples._data[:(m+5)][m][0]))
            sol = math.sqrt(ans / (samples._data[:(m+5)][m][0] - samples._data[:m][m-1][0]))
            finaldistances.append(sol)
    distanceMatrix = np.reshape(np.asarray(finaldistances), (10, 10))
    print(distanceMatrix)
    # print(Clustering(4, distanceMatrix))

    cl = Clustering(4, distanceMatrix)
    print(cl.medioids)
    print(cl.clusters)
#to compute the quaternions from the rotation rates obtained from the data








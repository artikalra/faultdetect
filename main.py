import util.filereader as rd
import math
import numpy as np
from data.cluster import Clustering

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")

    # for m in range(len(samples._data)):

    # Downsampling
    breakn = 10
    new_array = np.zeros((2000, 7))
    for i in range(len(new_array)):
        new_array[i, :] = np.mean(samples._data[i * breakn:(i + 1) * breakn - 1][:], 0)
        # print(new_array)
    samples._data = new_array
    print("downsampling done")

    """
    coeff = scipy.integrate.newton_cotes(len(samples._data))
    print(coeff)

    def integrate(function, a, b):
        ##coeff = [7, 32, 12, 32, 7]
        result = 0
        for i in range(0, len(coeff)):
            x = a + (i * (b - a)) / (len(coeff) - 1)
            result += coeff[i] * function(x)
        result = result * ((b - a) / sum(coeff))
        return result

    def func(x):
        #return x ** 0*distances
        return x ** 3 - 4 * x + 9

    print(integrate(func, -7.0, 7.0))
    """

    # to compute to integration of the square of L2norm for all the trajectories
    distances = np.zeros((1, len(samples._data) - 1))
    finaldistances = np.zeros((1, (len(samples._data) - 2) * (len(samples._data) - 2)))  # ((len(samples._data)-2)* (len(samples._data)-2))
    for m in range(1, len(samples._data) - 1):
        if m % 1000 == 0:
            print('dist ', m)
        for n in range(1, len(samples._data) - 1):
            # dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(n + 1)][n])])
            distances[:, m] = sum([(a - b) ** 2 for a, b in zip(samples._data[m - 1][4:7], samples._data[n-1][4:7])])
            # t = Symbol('t')
            # ans = integrate(dist, (t, samples._data[:m][m-1][0], samples._data[:(m+200)][m][0]))
            #print(distances)
            breakn = 10
            ans = sum(distances[m:m + breakn])
            finaldistances[:, m] = math.sqrt(ans / (samples._data[m][0] - samples._data[m - 1][0]))
    distanceMatrix = np.reshape((finaldistances), ((len(samples._data) - 2), (len(samples._data) - 2)))
    print(distanceMatrix)

    # to compute the quaternions from the rotation rates obtained from the data  and then take their logarithm
    # And to calculate the distance between the quaternions

    # from pyquaternion import Quaternion
    #
    # quatern_fault = np.zeros((4, len(samples._data) - 1))
    # quatern_dist = np.zeros((1, len(samples._data) - 1))
    # quatern_distances = np.zeros((1, (len(samples._data) - 2) * (len(samples._data) - 2)))
    # quatern_fault[:, 0] = [1., 0., 0., 0.]
    #
    # # h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
    # h = 0.2
    #
    # for i in range(1, len(samples._data) - 1):
    #     if i % 1000 == 0:
    #         print('quat ', i)
    #
    #     # quatern_fault [:,i]= samples.RK4((samples.KinematicModel(samples._data[:i][i-1][1:4], quatern_fault[:,0])), samples._data[:i][i-1][1:4], quatern_fault[:,i], h)
    #     function = samples.KinematicModel(samples._data[i - 1][1:4], quatern_fault[:, 0])
    #     quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[i - 1][1:4]),
    #                                       quatern_fault[:, i - 1], h)
    #     for n in range(1, i):
    #         # quatern_log[:, i] = Quaternion.log(Quaternion((quatern_fault[:,i])))
    #         quatern_dist[:, i] = Quaternion.distance(Quaternion((quatern_fault[:, i])), Quaternion((quatern_fault[:, n])))
    #         # t = Symbol('t')
    #         # ans = integrate(quatern_dist, (t, samples._data[:i][i-1][0], samples._data[:(i+200)][i][0]))
    #         breakn = 10
    #         ans = sum(quatern_distances[i:i + breakn])
    #         quatern_distances[:, i] = math.sqrt(ans / (samples._data[i][0] - samples._data[i - 1][0]))
    #
    # quatdistanceMatrix = np.reshape((quatern_distances), (len(samples._data) - 2, len(samples._data) - 2))
    #
    # for i in range(0, len(samples._data) - 2):
    #     for n in range(1, len(samples._data) - 2):
    #         if n == i:
    #             quatdistanceMatrix[i][n] = 0
    #         else:
    #             quatdistanceMatrix[i][n] = quatdistanceMatrix[n][i]

    final_matrix = distanceMatrix  #+ quatdistanceMatrix
    # print(final_matrix)

    cl = Clustering(4, final_matrix)
    print(cl.medioids)
    print(cl.clusters)

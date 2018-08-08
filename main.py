import util.filereader as rd
import math
import numpy as np

# from data.cluster import Clustering

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")

    # for m in range(len(samples._data)):

    # Downsampling
    breakn = 10
    new_array = np.zeros((100, 7))
    for i in range(len(new_array)):
        new_array[i, :] = np.mean(samples._data[i * breakn:(i + 1) * breakn - 1][:], 0)
        # print(new_array)
    samples._data = new_array
    print("downsampling done")
    print(len(samples._data))

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
    breakn = 10
    d = 0
    ssize = len(samples._data) - breakn
    finaldistances = np.zeros((ssize * ssize,))  # ((len(samples._data)-2)* (len(samples._data)-2))
    i = 0

    for m in range(1, ssize + 1):
        if m % 1000 == 0:
            print('dist ', m)
        for n in range(1, ssize + 1):
            d = 0.0
            for j in range(0, breakn):
                d += np.sum(
                    [(a - b) ** 2 for a, b in zip(samples._data[m - 1 + j][4:7], samples._data[n - 1 + j][4:7])])
            finaldistances[i] = math.sqrt(d)
            i += 1
    #   print(finaldistances[0:300])
    distanceMatrix = np.reshape((finaldistances), (ssize, ssize))
    #  print(distanceMatrix)

    # to compute the quaternions from the rotation rates obtained from the data  and then take their logarithm
    # And to calculate the distance between the quaternions

    from pyquaternion import Quaternion

    quatern_fault = np.zeros((4, len(samples._data) - 1))
    quatern_dist = np.zeros((1, len(samples._data) - 1))
    quatern_distances = np.zeros((1, (len(samples._data) - 2) * (len(samples._data) - 2)))
    quatern_fault[:, 0] = [1., 0., 0., 0.]

    # h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
    h = 0.2

    from pyquaternion import Quaternion

    q = 0
    breakn = 10
    ssize = len(samples._data) - breakn
    quatdistances = np.zeros((ssize * ssize,))
    quatern_fault = np.zeros((4, ssize + breakn))
    quatern_fault[:, 0] = [1., 0., 0., 0.]

    # # h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
    h = 0.2

    p = 0
    for i in range(1, ssize + breakn):
        quatern_fault[:, i] = samples.RK4(samples.KinematicModel, (samples._data[i - 1][1:4]),
                                          (quatern_fault[:, i - 1]), h)
    for i in range(1, ssize + 1):
        if i % 1000 == 0:
            print('quat ', i)
        # print(quatern_fault[:,i])
        for n in range(1, ssize + 1):
            q = 0.0
            # print(quatern_fault[:, n - 1])

            for j in range(0, breakn):
                # print(quatern_fault[:, i + j - breakn - 1 ])

                q += Quaternion.distance(Quaternion((quatern_fault[:, i + j - 1])),
                                         Quaternion((quatern_fault[:, n + j - 1])))
            quatdistances[p] = math.sqrt(q)
            p += 1
        quatdistanceMatrix = np.reshape((quatdistances), (ssize, ssize))
    print(quatdistanceMatrix)
    final_matrix = distanceMatrix + quatdistanceMatrix

    cl = Clustering(4, final_matrix)
    print(cl.medioids)
    with open('res.txt', 'w') as f:
        for p in cl.clusters:
            f.write(str(p[0]) + ',' + str(p[1]) + '\n')
    print(cl.clusters)

"""
    for i in range(0, len(samples._data) - 2):
        for n in range(1, len(samples._data) - 2):
            if n == i:
                quatdistanceMatrix[i][n] = 0
            else:
                quatdistanceMatrix[i][n] = quatdistanceMatrix[n][i]

    print(quatdistanceMatrix)
    """

# print(final_matrix)

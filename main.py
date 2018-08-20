import util.filereader as rd
import math
import numpy as np

from data.cluster import Clustering

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")

    #Calculation of spinnors
    from pyquaternion import Quaternion
    # quatern_fault = np.zeros((4,(len(samples._data)+1)))
    quatern_fault = np.zeros((4, len(samples._data)))
    spinnors = np.zeros((4, len(samples._data)))
    quatspinnors = np.zeros((3, len(samples._data)))

    quatern_fault[:, 0] = [1., 0., 0., 0.]
    h = 0.2

    for i in range(1, 11):
        if i % 100 == 0:
            print('spin ', i)
        quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                          quatern_fault[:, i - 1], h)
        my_quaternion = Quaternion(quatern_fault[:, i])
        spinnors[:,i]= Quaternion.log(my_quaternion)
        quatspinnors[:,i] = spinnors[1:4,i]
        #spinnors.append(log)
    print(quatspinnors.shape)
    print(np.array(samples._data).shape)
    samples._data = np.hstack((quatspinnors.T,np.array(samples._data)))
    print((samples._data).shape)

    ## Downsampling
    breakn = 10
    new_array = np.zeros((20669, 10))
    for i in range(len(new_array)):
        new_array[i, :] = np.mean(samples._data[i * breakn:(i + 1) * breakn - 1][:], 0)
        # print(new_array)
    samples._data = new_array
    print("downsampling done")
    print(len(samples._data))


#function to calculate the geodesic distance between two curves
    def geod_dim(gamma_i, gamma_f, m, dim):
        '''
         Developed by Alice Le Brigant
         Translated into Python by Murat Bronz

         Calcule la géodésique SRV entre deux courbes dans R3, càd le chemin de
         courbes qui relie les origines par une droite et qui interpole
         linéairement entre les  SRVF (vitesses renormalisées par la racine carrée
         de leur norme).

         Inputs :
         - gamma_i [dim x(n+1)] : courbe initiale
         - gamma_f [dim x(n+1)] : courbe finale
         - m : discrétisation en temps de la géodésique

         Outputss :
         - c [dim x(n+1)x(m+1)] : chemin de courbes géodésique de gamma_i à gamma_f
         - L : longueur de c = distance between gamma_i and gamma_f  '''
        n = len(gamma_i[0]) - 1
        T = np.linspace(0, 1, m + 1)
        taui = gamma_i[:, 1:n + 1] - gamma_i[:, 0:n]
        tauf = gamma_f[:, 1:n + 1] - gamma_f[:, 0:n]
        Ni = Nf = 0.0
        for i in range(dim):
            Ni += taui[i, :] ** 2
            Nf += tauf[i, :] ** 2
        Ni = Ni ** (1 / 4)
        Nf = Nf ** (1 / 4)
        # % Ni(Ni==0) = ones(size(Ni(Ni==0)));
        # % Nf(Nf==0) = ones(size(Nf(Nf==0)));
        qi = np.sqrt(n) * taui / np.tile(Ni, [dim, 1])  # 3 for 3 dimension, so change it to 6 for 6 dimnesions
        qf = np.sqrt(n) * tauf / np.tile(Nf, [dim, 1])
        TT = np.transpose(np.tile(np.tile(T, [dim, 1]), [n, 1, 1]), (1, 0, 2))
        A = np.transpose(np.tile(qi, [m + 1, 1, 1]), (1, 2, 0))
        B = np.transpose(np.tile(qf, [m + 1, 1, 1]), (1, 2, 0))
        q = A * (np.ones([dim, n, m + 1]) - TT) + B * TT  # .dot multiplication ?
        AAA = 0.0
        for i in range(dim):
            AAA += q[i, :, :] ** 2
        tau = 1 / n * np.transpose(np.tile(np.squeeze(AAA), [dim, 1, 1]) ** (1 / 2),
                                   [0, 1, 2]) * q  # tau=1/n*|q|*q # No need for transpose here for python ?
        c = np.zeros([dim, n + 1, m + 1])
        c[:, 0, :] = np.tile(np.ones([1, m + 1]) - T, [dim, 1]) * np.tile(gamma_i[:, 0], [m + 1, 1]).T + np.tile(T,
                                                                                                                 [dim,
                                                                                                                  1]) * np.tile(
            gamma_f[:, 0], [m + 1, 1]).T  # les origines sont reliées par une droite
        c[:, 1:n + 1, :] = tau
        c = np.cumsum(c, 1)  # check this dimension 1, in matlab it was 2...
        d1 = sum((gamma_f[:, 0] - gamma_i[:, 0]) ** 2)
        d2 = 1 / n * sum(sum((qf - qi) ** 2, 0))
        L = np.sqrt(d1 + d2)
        return L
    # for m in range(len(samples._data)):

    def symmetrize(matrix):
        return matrix + matrix.T

    # TO calculate distances between accelerometer components
    breakn = 10
    d = 0
    ssize = len(samples._data) - breakn
    finaldistances = np.zeros((ssize * ssize,))  # ((len(samples._data)-2)* (len(samples._data)-2))
    i = 0

    for m in range(1, ssize + 1):
        if m % 100 == 0:
            print('dist ', m)
        c1 = (samples._data[m - 1:m + breakn - 1, 4:10]).T
        # print(c1.shape)
        for n in range(1, ssize + 1):
            if n < m:
                c2 = (samples._data[n - 1:n + breakn - 1, 4:10]).T
                d = geod_dim(c1, c2, 1, 6)
                finaldistances[i] = d
            i += 1
    distanceMatrix = symmetrize(np.reshape((finaldistances), (ssize, ssize)))
    print(distanceMatrix)

    cl = Clustering(4, distanceMatrix)
    print(cl.medioids)
    with open('results.txt', 'w') as f:
        for p in cl.clusters:
            f.write(str(p[0]) + ',' + str(p[1]) + '\n')
    print(cl.clusters)

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
    """
    # to compute to integration of the square of L2norm for all the trajectories
    breakn = 10
    d = 0
    ssize = len(samples._data) - breakn
    finaldistances = np.zeros((ssize * ssize,))  # ((len(samples._data)-2)* (len(samples._data)-2))
    i = 0

    for m in range(1, ssize + 1):
        if m % 100 == 0:
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


    # to compute the quaternions from the rotation rates obtained from the data  and then take their logarithm
    # And to calculate the Euclidean distance between the quaternions

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
        if i % 100 == 0:
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

    #print(final_matrix)
 """



"""
    #TO calculate distances between accelerometer components
    breakn = 10
    d = 0
    ssize = len(samples._data) - breakn
    finaldistances = np.zeros((ssize * ssize,))  # ((len(samples._data)-2)* (len(samples._data)-2))
    i = 0

    for m in range(1, ssize + 1):
        if m % 100 == 0:
            print('dist ', m)
        c1 = (samples._data[m-1:m + breakn-1, 4:7]).T
        #print(c1.shape)
        for n in range(1, ssize + 1):
            if n < m:
                c2 = (samples._data[n-1:n+breakn-1, 4:7]).T
                d = geod_dim(c1, c2, 1, 3)
                finaldistances[i] = d
            i += 1
    distanceMatrix = symmetrize(np.reshape((finaldistances), (ssize, ssize)))
    print(distanceMatrix)

    # TO calculate distances between spinnors
    from pyquaternion import Quaternion
    q = 0
    breakn = 10
    ssize = len(samples._data) - breakn
    quatdistances = np.zeros((ssize * ssize,))
    quatern_fault = np.zeros((4, ssize + breakn))
    spinnors = np.zeros((4, ssize + breakn))
    quatern_fault[:, 0] = [1., 0., 0., 0.]

    # # h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
    h = 0.2
    p = 0
    for i in range(1, ssize + breakn):
        quatern_fault[:, i] = samples.RK4(samples.KinematicModel, (samples._data[i - 1][1:4]),
                                          (quatern_fault[:, i - 1]), h)
        quat = Quaternion(quatern_fault[:, i])
        spinnors[:,i]= Quaternion.log(quat)


    for i in range(1, ssize + 1):
        if i % 100 == 0:
            print('quat ', i)
        c1 = (spinnors[1:4,i - 1:i + breakn - 1])
        for n in range(1, ssize + 1):
            if n < i:
                c2 = (spinnors[1:4,n - 1:n + breakn - 1])
                q = geod_dim(c1, c2, 1, 3)
                quatdistances[p] = q
            p += 1
    quatdistanceMatrix = symmetrize(np.reshape((quatdistances), (ssize, ssize)))
    print(quatdistanceMatrix)

    finalMatrix=distanceMatrix + quatdistanceMatrix
    print(finalMatrix)

"""





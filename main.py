import util.filereader as rd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import odeint


from data.cluster import Clustering

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD_162k-168k.data")

    # Calculation of spinnors
    from pyquaternion import Quaternion
    quatern_spin = np.zeros((3, len(samples._data)))
    quatern_fault = np.zeros((4, len(samples._data)))
    quatspinnors = np.zeros((3, len(samples._data)))
    quatern_spin[:, 0] = [0.00954169, 0.01104921, 0.00936013]
    #quatern_fault[:, 0] = [0.99984563, 0.00954117, 0.01104861, 0.00935962]
    quatern_fault[:, 0] = [1.0, 0.0, 0.0, 0.0]
    h = 0.0167
    gyro = np.array(samples._data)[:, 1:4]





# should paste quaternion function here

    breakn=10
    ssize = len(samples._data) - breakn
    qua = np.zeros((4,breakn,ssize))
    qua[:, 0, 0] = [1.0, 0.0, 0.0, 0.0]
    for i in range(1, ssize+1):
        if i % 100 == 0:
            print('spin ', i)
        for j in range(0, breakn):
            #print(np.array(samples._data[:i + j +1][i + j][0]))
            #print(np.array(samples._data[:i+j] [i - 1 + j][1:4]))
            #print(qua[:,j, i-1])
            qua[: , j , i-1]=(odeint(samples.KinematicModel, qua[:,0, 0], [np.array(samples._data[:i+j][i - 1 + j][0]),np.array(samples._data[:i+j+1][i+j][0])], args=(np.array(samples._data[:i+j][i - 1 + j][1:4]),)))[1,:]
            #print(qua)
           # my_quaternion = Quaternion(quatern_fault[:, i])
        #    quatspinnors[:, i] = Quaternion.log(my_quaternion).elements[1:4]
    samples._data = np.array(samples._data)
    print((samples._data).shape)
    #samples._data = np.hstack((np.array(samples._data), quatspinnors.T))
    #print((samples._data).shape)

    """
    figure = plt.figure()
    plt.plot(samples._data[:, 0], samples._data[:, 11])
    #ax.axvspan(100 , 2000, facecolor='g', alpha=0.5)
    #ax.axvspan(2000, 3000, facecolor='b', alpha=0.5)
    plt.show()

    figure = plt.figure()
    plt.plot(samples._data[:, 0], samples._data[:, 10])
    plt.show()
    """
    """
    # Calculation of spinnors
    quatern_spin = np.zeros((3, len(samples._data)))
    #spinnors = np.zeros((4, len(samples._data)))
    #quatspinnors = np.zeros((3, len(samples._data)))
    quatern_spin[:, 0] = [0.0853,   0.1004,   0.0833]
    h = 0.2

    for i in range(1, 10000):
        if i % 100 == 0:
            print('spin ', i)
        quatern_spin[:, i] = samples.RK4(samples.KinematicModel2, np.array(samples._data[:i][i - 1][1:4]),
                                          quatern_spin[:, i - 1], h)
        #print("quatspin", quatern_spin)
    samples._data = np.hstack((np.array(samples._data), quatern_spin.T))
    print((samples._data).shape)
    #print(samples._data)

    # qlog1 = Quaternion.log(Quaternion([0.988,0.085,0.100,0.083]))
    # qlog2 = Quaternion.log(Quaternion([1.0, 0.0 , 0.0, 0.0]))
    # print(qlog1, qlog2)
    # print(Quaternion([0.988,0.085,0.100,0.083]))

    
        #Calculation of spinnors
        from pyquaternion import Quaternion
        # quatern_fault = np.zeros((4,(len(samples._data)+1)))
        quatern_fault = np.zeros((4, len(samples._data)))
        spinnors = np.zeros((4, len(samples._data)))
        quatspinnors = np.zeros((3, len(samples._data)))

        quatern_fault[:, 0] = [1., 0., 0., 0.]
        h = 0.2

        for i in range(1, 10):
            if i % 100 == 0:
                print('spin ', i)
            quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                              quatern_fault[:, i - 1], h)
            print(quatern_fault.shape)
            my_quaternion = Quaternion(quatern_fault[:, i])
            #print(my_quaternion)
            quatspinnors[:, i] = Quaternion.log(my_quaternion).elements[1:4]
            print("quatspinnors",quatspinnors)

            #quatspinnors[:, i] = spinnors[1:4,i]
            #print(quatspinnors.shape)
        samples._data = np.hstack((np.array(samples._data),quatspinnors.T))
        print((samples._data).shape)
        #print(samples._data)
    """

    ## Downsampling
    breakn = 10
    nominal = np.array([1.0, 1.0, 0.0, 0.0])
    #print((samples._data).shape)
    new_array = np.zeros((600, len(samples._data[0, :])))
    #print(new_array)
    for i in range(len(new_array)):


        new_array[i, :] = np.mean(np.concatenate((samples._data[i * breakn:(i + 1) * breakn - 1][0:7],
                                                  samples._data[i * breakn:(i + 1) * breakn - 1][11:14])), 0)
        # print(nominal in samples._data[i * breakn:(i + 1) * breakn - 1 , 7:11])
        # print(np.array(samples._data[i * breakn:(i + 1) * breakn - 1, 7:11]) != nominal)

        if nominal in samples._data[i * breakn:(i + 1) * breakn - 1, 7:11]:
            # print("no")
            # if samples._data[i * breakn:(i + 1) * breakn - 1][7:11] != nominal:
            if np.any(samples._data[i * breakn:(i + 1) * breakn - 1, 7:11] != nominal):
                # print("nono")
                new_array[i, 7:11] = np.array(
                    [2, 2, 2, 2])  # The transition case T=2 (where we have both nominal and faulty cases)
            new_array[i, 7:11] = np.array([0, 0, 0, 0])  # Nominal Case N=0
        else:
            # print("yes")
            new_array[i, 7:11] = np.array([1, 1, 1, 1])  # Faulty Case F=1
        print(new_array[i, 7:11])


    samples._data = new_array.copy()
    # print(samples._data)
    print("downsampling done")
    print(np.array(samples._data).shape)


    # figure = plt.figure()
    # plt.plot(samples._data[:, 0], samples._data[:, 11])
    # plt.show()

    # function to calculate quaternions corresponding to each sliding window
    def calcquaternion(data):
        breakn = 10
        ssize = len(data) - breakn
        qua = np.zeros((4, breakn, ssize))
        qua[:, 0, 0] = [1.0, 0.0, 0.0, 0.0]
        for i in range(1, ssize + 1):
            if i % 100 == 0:
                print('spin ', i)
            for j in range(0, breakn):
                qua[:, j, i - 1] = (odeint(samples.KinematicModel, qua[:, 0, 0],
                                           [(data[i - 1 + j, 0]),
                                            (data[i + j, 0])],
                                           args=((data[i - 1 + j, 1:4]),)))[1, :]
        return qua


    def calcspinnors(data):
        breakn = 10
        ssize = len(data) - breakn
        qua = np.zeros((4, breakn, ssize))
        spinnor = np.zeros((3, breakn, ssize))
        qua[:, 0, 0] = [1.0, 0.0, 0.0, 0.0]
        for i in range(1, ssize + 1):
            # if i % 100 == 0:
            #     print('calculatingspin ', i)
            for j in range(0, breakn):
                qua[:, j, i - 1] = (odeint(samples.KinematicModel, qua[:, 0, 0],
                                           [(data[i - 1 + j, 0]),
                                            (data[i + j, 0])],
                                           args=((data[i - 1 + j, 1:4]),)))[1, :]

                my_quaternion = Quaternion(qua[:, j, i - 1])
                spinnor[:,j, i-1] = Quaternion.log(my_quaternion).elements[1:4]
        return spinnor



    # function to calculate the L2distance by updating the quaternions in each siding window
    def L2slidingdist(data, start, stop):
        breakn = 1  # This is the sliding window size
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize, ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0
        spinnor = calcspinnors(data)
        for m in range(start, stop):  # start limit=1 stop limit ssize+1

            if m % 100 == 0:
                print('dist ', m)

            for n in range(1, ssize + 1):
                d = 0.0
                # my_quaternion = Quaternion(qua[:,:,i])
                # spinnor = Quaternion.log(my_quaternion).elements[1:4]
                for j in range(0, breakn):
                    #spinnor = calcspinnors(data)
                    #print(spinnor.shape)
                    c1 = np.concatenate(((data[m - 1 + j, 4:7]), (spinnor[:, j-1, m - 1 + j-10])))
                    c2 = np.concatenate(((data[n - 1 + j, 4:7]), (spinnor[:, j-1, n - 1 + j-10])))
                    d += np.sum([(a - b) ** 2 for a, b in zip(c1, c2)])
                finaldistances[m - 1][n - 1] = math.sqrt(d)
                i += 1
        distanceMatrix = symmetrize((finaldistances))
        return distanceMatrix

#function to calculate the L2distance by updating the quaternions in each siding window
    def L2dist(data, start, stop):
        breakn = 1  # This is the sliding window size
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize, ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(start, stop):  # start limit=1 stop limit ssize+1
            if m % 100 == 0:
                print('dist ', m)
            for n in range(1, ssize+1):
                d = 0.0
                for j in range(0, breakn):
                    quatern_fault[:,i] = (odeint(samples.KinematicModel, quatern_fault[:, i - 1],
                                                  [(data[m - 1 + j,0]),(data[m + j,0])],
                                                  args=((data[m - 1 + j, 4:7]),)))[1, :]
                    my_quaternion = Quaternion(quatern_fault[:,i])
                    spinnor = Quaternion.log(my_quaternion).elements[1:4]
                    c1 = np.concatenate(((data[m - 1 + j, 4:7]), (spinnor[m - 1 + j, :])))
                    c2 = np.concatenate(((data[n - 1 + j, 4:7]), (spinnor[n - 1 + j, :])))
                    d += np.sum([(a - b) ** 2 for a, b in zip(c1, c2)])
                finaldistances[m - 1][n - 1] = math.sqrt(d)
                i += 1
        distanceMatrix = symmetrize((finaldistances))
        return distanceMatrix

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

    # to compute to integration of the square of L2norm for all the trajectories


    def symmetrize(matrix):
        return matrix + matrix.T


    def distance(data, start, stop):
        breakn = 10
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize , ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(start, stop):
            if m % 100 == 0:
                print('dist ', m)
            c1 = np.concatenate(
                ((data[m - 1:m + breakn - 1, 4:7]).T, (data[m - 1:m + breakn - 1, 11:14]).T))
            # print(c1.shape)
            for n in range(1, ssize + 1):
                if n < m:
                    c2 = np.concatenate(
                        ((data[n - 1:n + breakn - 1, 4:7]).T, (data[n - 1:n + breakn - 1, 11:14]).T))
                    d = geod_dim(c1, c2, 1, 6)
                    finaldistances[m-1][n-1] = d
                i += 1
        distanceMatrix = symmetrize((finaldistances))
        return distanceMatrix


    def L2distance(data, start, stop):
        breakn = 1  # This is the sliding window size
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize, ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(start, stop):  # start limit=1 stop limit ssize+1
            if m % 100 == 0:
                print('dist ', m)
            for n in range(1, ssize+1):
                d = 0.0
                for j in range(0, breakn):
                    c1 = np.concatenate(((data[m - 1 + j, 4:7]), (data[m - 1 + j, 11:14])))
                    c2 = np.concatenate(((data[n - 1 + j, 4:7]), (data[n - 1 + j, 11:14])))
                    d += np.sum([(a - b) ** 2 for a, b in zip(c1, c2)])
                finaldistances[m - 1][n - 1] = math.sqrt(d)
                i += 1
        distanceMatrix = symmetrize((finaldistances))
        return distanceMatrix

    breakn = 1
    distanceMatrix = L2slidingdist((samples._data), 1,(len(samples._data)+1- breakn ) )  #(len(samples._data) + 1 - breakn)
    print(distanceMatrix.shape)


    cl = Clustering(2, distanceMatrix)
    print(cl.medioids)
    with open('results.txt', 'w') as f:
        for p in cl.clusters:
            f.write(str(p[0]) + ',' + str(p[1]) + '\n')
    print(cl.clusters)


# To plot the pixel graph

    pickle.dump(distanceMatrix, open("save.p", "wb"))
    distance_matrix = pickle.load(open("save.p", "rb"))
    diff = distanceMatrix - distance_matrix
    print("difference of the matrices:" , diff)
    print(distance_matrix)
    plt.matshow(distance_matrix, cmap=plt.cm.gray)
    plt.legend()
    plt.show()

    # To plot the results directly using the text file
    filename = './results.txt'
    R = []
    with open(filename, 'r') as f:
        i = 0
        for line in f:
            field = line.split(',')
            t = int(field[0])
            c = int(field[1])
            R.append((t, c))
            i += 1
    Rn = np.array(R)

    figure = plt.figure()
    plt.plot(Rn[:, 0], Rn[:, 1], "o")
    plt.show()

    """
    
    
    # calculation of spinnors
        for i in range(1, len(samples._data)):
        if i % 100 == 0:
            print('spin ', i)
        quatern_spin[:, i] = samples.RK4(samples.KinematicModel2, np.array(samples._data[:i][i - 1][1:4]),
                                         quatern_spin[:, i - 1], h)
        #quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                          #quatern_fault[:, i - 1], h)
        quatern_fault[:,i]=(odeint(samples.KinematicModel, quatern_fault[:, i-1], [np.array(samples._data[:i][i - 1][0]),np.array(samples._data[:i+1][i][0])], args=(np.array(samples._data[:i][i - 1][1:4]),)))[1,:]
        my_quaternion = Quaternion(quatern_fault[:, i])
        quatspinnors[:, i] = Quaternion.log(my_quaternion).elements[1:4]
    samples._data = np.hstack((np.array(samples._data), quatspinnors.T))
    print((samples._data).shape)
    
    
        # TO calculate distances between accelerometer and quaternion functions components
        breakn = 10
        d = 0
        ssize = len(samples._data) - breakn
        finaldistances = np.zeros((ssize * ssize,))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(1, ssize + 1):
            if m % 100 == 0:
                print('dist ', m)

            c1 = np.concatenate(((samples._data[m - 1:m + breakn - 1, 4:7]).T,(samples._data[m - 1:m + breakn - 1, 11:14]).T))
            #print(c1.shape)
            for n in range(1, ssize + 1):
                if n < m:
                    c2 = np.concatenate(((samples._data[n - 1:n + breakn - 1, 4:7]).T,(samples._data[n - 1:n + breakn - 1, 11:14]).T))
                    d = geod_dim(c1, c2, 1, 6)
                    finaldistances[i] = d
                i += 1
        distanceMatrix = symmetrize(np.reshape((finaldistances), (ssize, ssize)))
        #print(distanceMatrix)
    """

"""
# to plot using the clusters
    x = np.zeros((1, len(cl.clusters)))
    y = np.zeros((1, len(cl.clusters)))
    for i in range(len(cl.clusters)):
        x[:,i] =(cl.clusters)[i][0]
    #print(x)

    for j in range(len(cl.clusters)):
        y[:, j] = (cl.clusters)[j][1]
    #print(y)

    plt.plot(x, y, 'ro')
    plt.axis([0, 100, 0, 5])
    plt.show()


    
   


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
    h = 0.167

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
        print(spinnors)



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





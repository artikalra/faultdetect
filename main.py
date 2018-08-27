import util.filereader as rd
import math, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import tempfile

from data.cluster import Clustering

from multiprocessing import Process, JoinableQueue
import threading 
import os

def info(title):
    print(title)
    print('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print('parent process:', os.getppid())
    print('process id:', os.getpid())


class SummingThread(threading.Thread):
    def __init__(self,data,start_indx,stop_indx):
        super(SummingThread, self).__init__()
        self.data = data
        self.start_indx=start_indx
        self.stop_indx=stop_indx
        self.total=0

    def run(self):
        # for i in range(self.start_indx,self.stop_indx):
        #     self.total+=i
        self._distance(self.data, self.start_indx, self.stop_indx)

    def _distance(self, data, start_indx, stop_indx):
        breakn = 10
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize , ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(start_indx, stop_indx):
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
        self.distance_matrix = finaldistances
        
def Factorizer(data, nprocs):
    def Process_distance( data, start_indx, stop_indx, out_q):
        breakn = 10
        d = 0
        ssize = len(data) - breakn
        finaldistances = np.zeros((ssize , ssize), dtype=float)  # ((len(samples._data)-2)* (len(samples._data)-2))
        i = 0

        for m in range(start_indx, stop_indx):
            if m % 100 == 0:
                print('dist ', m)

            c1 = np.concatenate(
                ((data[(m-1):(m+breakn-1), 4:7]).T, (data[(m-1):(m+breakn-1), 11:14]).T))
            for n in range(1, ssize + 1):
                if n < m:
                    c2 = np.concatenate(
                        ((data[n - 1:n + breakn - 1, 4:7]).T, (data[n - 1:n + breakn - 1, 11:14]).T))
                    d = geod_dim(c1, c2, 1, 6)
                    # print(d)
                    finaldistances[m-1,n-1] = d
                i += 1
        # print(finaldistances)
        # Random file generator of python
        tf = tempfile.NamedTemporaryFile(prefix='tmp_', dir='./', delete=False)
        with open(tf.name, 'wb') as f:
            np.save(f, finaldistances)
        # out_q.put(finaldistance)

    ssize = len(data) - 10 #breakn
    nr_elements = int(ssize/(nprocs-1))
    procs_list = []
    distanceMatrix = np.zeros((ssize, ssize), dtype=float)

    # start_time = time.time()

    # Distribute the job to threads
    out_q = JoinableQueue()
    # for rep in range(1):
    for i in range(nprocs-1):
        p = Process(target=Process_distance,
                args=(data, (i*nr_elements), (i+1)*nr_elements, out_q) )
        procs_list.append(p)
        p.start()
        print(p.is_alive())

    p = Process(target=Process_distance,
                args=(data, ((nprocs-1)*nr_elements), ssize, out_q) )
    procs_list.append(p)
    p.start()

    print(procs_list)
    print('Here I am to join !')
    for p in procs_list:
        p.join()
        print(p.is_alive())
        
        # if not any(p.is_alive() for p in r) and out_queue.empty():

        # print('Here I am to get() !')
        # i =0
        # while out_q.empty() == False:
        #     distanceMatrix += out_q.get() #False works with 300 data...
        #     print(i, distanceMatrix)
        #     i += 1
        # # for p in procs_list:
        # #     p.()
        # proc_list = []
        # print(procs_list)

    return distanceMatrix



if __name__ == '__main__':
    # info('main line')
    fname = "17_07_06__10_21_07_SD_small.csv"
    if os.path.isfile(fname):
        df = pd.read_csv(fname)
        data = df.values #df.reset_index().values

    else:
        samples = rd.parse("data/17_07_06__10_21_07_SD.data")

        #Calculation of spinnors
        from pyquaternion import Quaternion
        # quatern_fault = np.zeros((4,(len(samples._data)+1)))
        quatern_fault = np.zeros((4, len(samples._data)))
        # spinnors = np.zeros((4, len(samples._data)))
        quatspinnors = np.zeros((3, len(samples._data)))

        quatern_fault[:, 0] = [1., 0., 0., 0.]
        h = 0.16667 # 60Hz 

        for i in range(1, (len(samples._data))):
            if i % 100 == 0:
                print('spin ', i)
            quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                              quatern_fault[:, i - 1], h)
            my_quaternion = Quaternion(quatern_fault[:, i])
            # spinnors[:,i]= Quaternion.log(my_quaternion)
            # quatspinnors[:,i] = spinnors[1:4,i]
            quatspinnors[:,i] = Quaternion.log(my_quaternion).elements[1:4]
        data = np.hstack((np.array(samples._data),quatspinnors.T))
        #print((samples._data).shape)
        #print(samples._data)

        # Record the processed data into csv for faster next run
        df = pd.DataFrame(data, columns=['time', 'G1', 'G2', 'G3', 'A1', 'A2', 'A3', 'F1', 'F2', 'F3', 'F4', 'S1', 'S2', 'S3'])
        df.to_csv("17_07_06__10_21_07_SD.csv", index=None)


    ## Downsampling
    breakn = 10
    nominal = np.array([1.0, 1.0, 0.0, 0.0])
    new_array = np.zeros((int(len(data)/10), len(data[0,:]))) #20669
    for i in range(len(new_array)):
        new_array[i, :] = np.mean(np.concatenate((data[i * breakn:(i + 1) * breakn - 1][0:7],data[i * breakn:(i + 1) * breakn - 1][11:14])), 0)
        #print(nominal in data[i * breakn:(i + 1) * breakn - 1 , 7:11])
        #print(np.array(data[i * breakn:(i + 1) * breakn - 1, 7:11]) != nominal)
        if nominal in data[i * breakn:(i + 1) * breakn - 1 , 7:11]:
            #print("no")
            #if data[i * breakn:(i + 1) * breakn - 1][7:11] != nominal:
            if  np.any(data[i * breakn:(i + 1) * breakn - 1 ,7:11] != nominal):
                #print("nono")
                new_array[i, 7:11] = np.array([2, 2, 2, 2]) # The transition case T=2 (where we have both nominal and faulty cases)
            new_array[i, 7:11] = np.array([0, 0, 0, 0]) #Nominal Case N=0
        else:
            #print("yes")
            new_array[i, 7:11] = np.array([1, 1, 1, 1]) # Faulty Case F=1

    data = new_array

    # Record the downsampled data into csv for faster next run as well
    df = pd.DataFrame(data)
    df.to_csv("17_07_06__10_21_07_SD_small_ds.csv", header=None)
    #print(data)
    print("downsampling done")
    print(np.array(data).shape)


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


    # def distance(data, start, stop):
    #     breakn = 10
    #     d = 0
    #     ssize = len(data) - breakn
    #     finaldistances = np.zeros((ssize , ssize))  # ((len(samples._data)-2)* (len(samples._data)-2))
    #     i = 0

    #     for m in range(start, stop):
    #         if m % 100 == 0:
    #             print('dist ', m)

    #         c1 = np.concatenate(
    #             ((data[m - 1:m + breakn - 1, 4:7]).T, (data[m - 1:m + breakn - 1, 11:14]).T))
    #         # print(c1.shape)
    #         for n in range(1, ssize + 1):
    #             if n < m:
    #                 c2 = np.concatenate(
    #                     ((data[n - 1:n + breakn - 1, 4:7]).T, (data[n - 1:n + breakn - 1, 11:14]).T))
    #                 d = geod_dim(c1, c2, 1, 6)
    #                 finaldistances[m-1][n-1] = d
    #             i += 1
    #     distanceMatrix = symmetrize((finaldistances))
    #     return distanceMatrix

    #distanceMatrix = distance((samples._data), 1, (len(samples._data)+1-breakn))
    
    #data = samples._data
    
    # ssize = len(data) - 10 #breakn
    # dsize = len(data)
    # nr_procs = 3
    # nr_elements = 22 #int(ssize/nr_threads)
    # procs_list = []
    # queue_list = []
    # distanceMatrix = np.zeros((ssize, ssize), dtype=float)

    # start_time = time.time()
    # # Distribute the job to threads

    # out_q = Queue()

    # for i in range(nr_procs):
    #     # th = SummingThread(data, (i*nr_elements)-nr_elements, i*nr_elements)

    #     p = Process(target=Process_distance, args=(data, (i*nr_elements), (i+1)*nr_elements, out_q) )
    #     procs_list.append(p)
    #     p.start()

    # # # Addint the last bit of the data
    # # th = SummingThread(data, nr_elements*nr_threads, ssize)
    # # thread_list.append(th)

    # for i in range(nr_procs):
    #     print(out_q.get())

    # for p in procs_list:
    #     p.join()

    # # for thread in thread_list:
    # #     distanceMatrix += thread.finaldistances #thread.distance_matrix
    # # i = 0
    # # for q in queue_list:
    # #     distanceMatrix += q.get()
    # #     print(i, q.get())
    # #     i += 1

    # distanceMatrix = symmetrize(distanceMatrix)

    # th1 = SummingThread(data, 0, 100)
    # print('Created the thread 1')
    # th1.start()

    # th1.join()
    start_time = time.time()
    # print('Here is the 1st thread : ',th1.distance_matrix)
    # distanceMatrix = Factorizer(data, 4)

    Factorizer(data, 4)
    distanceMatrix = symmetrize(distanceMatrix)
    duration = time.time()-start_time
    print('Duration : ', duration)
    print('Here I am ! 2 ')
    cl = Clustering(4, distanceMatrix)
    # print(cl.medioids)
    print(distanceMatrix)

    # A = np.array(distanceMatrix)
    # with open(r"dist_mat.pkl", "wb") as output_file:
    #     pickle.dump(distanceMatrix, output_file)

    with open("distancematrix.txt", 'wb') as f:
        np.save(f, distanceMatrix)

    with open('clustered.txt', 'w') as f:
        for p in cl.clusters:
            f.write(str(p[0]) + ',' + str(p[1]) + '\n')
    print(cl.clusters)


# To plot the pixel graph

    # pickle.dump(distanceMatrix, open("save.p", "wb"))

    # distance_matrix = pickle.load(open("save.p", "rb"))
    # diff = distanceMatrix - distance_matrix
    # print("difference of the matrices:" , diff)
    # print(distance_matrix)
    plt.figure()
    plt.matshow(distanceMatrix, cmap=plt.cm.gray)
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
# def get_timed_interruptable(q, timeout):
#     stoploop = time.monotonic() + timeout - 1
#     while time.monotonic() < stoploop:
#         try:
#             return q.get(timeout=1)  # Allow check for Ctrl-C every second
#         except queue.Empty:
#             pass
#     # Final wait for last fraction of a second
#     return q.get(timeout=max(0, stoploop + 1 - time.monotonic())) 




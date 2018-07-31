import util.filereader as rd
import math
import numpy as np
from data.cluster import Clustering



if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")

    #for m in range(len(samples._data)):
    """
    x=1
    y=11
    smallsample = []
    mydata = np.zeros((20,4))
    print (len(mydata))
   
    breakn = 10
    new_array=np.zeros((2,4))
    for i in range(2):
        print (i)
        #print(mydata[i*breakn:(i+1)*breakn-1])
        new_array.append(np.mean(mydata[i*breakn:(i+1)*breakn-1][:], 0))
    print (new_array)
    """
    breakn = 10
    new_array=[]
    for i in range(20600):
        #print (i)
        new_array.append(np.mean(samples._data[i*breakn:(i+1)*breakn-1][:], 0))
    #print(np.array(new_array))

    """
    for n in range(1,20):
        for m in range(x, y):
            ans =np.mean((np.array(samples._data[:m][m-1])), axis=0)
        x=y
        y=x+10
        smallsample.append(ans)
    print((smallsample))
    ###
    """
    samples._data = new_array


    #print(len(samples._data[::10]))



# to compute to integration of the square of L2norm for all the trajectories
    from sympy import *
    finaldistances = []
    for m in range(1, len(samples._data)-1):
        for n in range(1, len(samples._data)-1):
            # dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(n + 1)][n])])
            #print(samples._data[:m][m-1][4:7])
            #print(samples._data[:n][n - 1][4:7])#to use only for the accelerometer components
            dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1][4:7], samples._data[:n][n - 1][4:7])])
            t = Symbol('t')
            ans = integrate(dist, (t, samples._data[:m][m-1][0], samples._data[:(m+200)][m][0]))
            sol = math.sqrt(ans / (samples._data[:(m+200)][m][0] - samples._data[:m][m-1][0]))
            finaldistances.append(sol)
    distanceMatrix = np.reshape(np.asarray(finaldistances), ((len(samples._data)-2), (len(samples._data)-2)))
    #print(distanceMatrix)


    #cl = Clustering(4, distanceMatrix)
    #print(cl.medioids)
    #print(cl.clusters)
    #print(len(samples._data))

#to compute the quaternions from the rotation rates obtained from the data  and then take their logarithm
# And to calculate the distance between the quaternions


    from pyquaternion import Quaternion

    #quatern_fault = np.zeros((4,(len(samples._data)+1)))
    #quatern_fault = np.zeros((4, 11))
    quatern_fault = np.zeros((4, len(samples._data)-1))
    quatern_log = []
    #quatern_dist = np.zeros((5,))
    quatern_dist = []
    quat_dist =[]

    quatern_fault[:,0] = [1.,0.,0.,0.]
    # h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
    h = 0.2



    #print(len(samples._data))
    # for i in range(1,len(samples._data)-1):
    for i in range(1, len(samples._data)-1):
        for n in range(1, len(samples._data)-1):
            # quatern_fault [:,i]= samples.RK4((samples.KinematicModel(samples._data[:i][i-1][1:4], quatern_fault[:,0])), samples._data[:i][i-1][1:4], quatern_fault[:,i], h)
            function = samples.KinematicModel(samples._data[:i][i - 1][1:4], quatern_fault[:, 0])
            quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                              quatern_fault[:, i - 1], h)

            #my_quaternion = Quaternion(quatern_fault[:, i])
            #log = Quaternion.log(my_quaternion)
            #quatern_log.append(log)



            if n < i :
            # quatern_log[:, i] = Quaternion.log(Quaternion((quatern_fault[:,i])))
                quatern_dist = Quaternion.distance(Quaternion((quatern_fault[:, i])), Quaternion((quatern_fault[:, n])))

                t = Symbol('t')
                ans = integrate(quatern_dist, (t, samples._data[:i][i-1][0], samples._data[:(i+200)][i][0]))
                sol = math.sqrt(ans / (samples._data[:(i+200)][i][0] - samples._data[:i][i-1][0]))

            quat_dist.append(sol)

    quatdistanceMatrix = np.reshape(np.asarray(quat_dist), (len(samples._data)-2, len(samples._data)-2))


    for i in range(0,len(samples._data)-2):
        for n in range (1,len(samples._data)-2):
            if n == i:
                quatdistanceMatrix[i][n] = 0
            else:
                quatdistanceMatrix[i][n] = quatdistanceMatrix[n][i]



    #print(quatdistanceMatrix)
    #quatdistanceMatrix[i][i+1]= quatdistanceMatrix[i][i-1]
    #print(quatdistanceMatrix)
    # print(quat_dist)

    final_matrix = distanceMatrix + quatdistanceMatrix
    #print(final_matrix)

    cl = Clustering(4, final_matrix)
    print(cl.medioids)
    print(cl.clusters)

    # list_log = np.ndarray.tolist(log.vector)

    # dist = sum([(a - b) ** 2 for a, b in zip(list_log[:i][0:3], list_log[:n][0:3])])
    # t = Symbol('t')
    # ans = integrate(dist, (t,samples._data[:i][i - 1][0], samples._data[:(i + 5)][i][0]))
    # sol = math.sqrt(ans / (samples._data[:(i + 5)][i][0] - samples._data[:i][i- 1][0]))
    # quatern_dist.append(sol)

    # quatdistanceMatrix = np.reshape(np.asarray(quatern_dist), (4, 4))

    # print((quatern_fault))
    # print(quatern_log)
    # print(quatdistanceMatrix)

    """
    # for i in range(1,len(samples._data)-1):
    for i in range(1, 5):
        #for n in range(1,5):
            # quatern_fault [:,i]= samples.RK4((samples.KinematicModel(samples._data[:i][i-1][1:4], quatern_fault[:,0])), samples._data[:i][i-1][1:4], quatern_fault[:,i], h)
            function = samples.KinematicModel(samples._data[:i][i - 1][1:4], quatern_fault[:, 0])
            quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                              quatern_fault[:, i - 1], h)

            my_quaternion = Quaternion(quatern_fault[:, i])
            log = Quaternion.log(my_quaternion)
            quatern_log.append(log.vector)

            # quatern_log[:, i] = Quaternion.log(Quaternion((quatern_fault[:,i])))
            #quatern_dist = Quaternion.distance(Quaternion((quatern_fault[:, i])), Quaternion((quatern_fault[:, n])))
            #quat_dist.append(quatern_dist)
            # quatdistanceMatrix = np.reshape(np.asarray(quat_dist), (4, 4))
            # print(quatdistanceMatrix)
            # print(quat_dist)

            list_log = np.ndarray.tolist(log.vector)
            #print(quatern_log[:i][i-1][0:3])
            for n in range(1, 5):
                #print(quatern_log[:i][i - 1][0:3])
                print(quatern_log[:,n])

                #dist = sum([(a - b) ** 2 for a, b in zip(quatern_log[:i][i-1][0:3], quatern_log[:n][n-1][0:3])])
                #t = Symbol('t')
                #ans = integrate(dist, (t, samples._data[:i][i - 1][0], samples._data[:(i + 5)][i][0]))
                #sol = math.sqrt(ans / (samples._data[:(i + 5)][i][0] - samples._data[:i][i - 1][0]))
                #quatern_dist.append(dist)
    #print(quatern_dist)
    #quatdistanceMatrix = np.reshape(np.asarray(quatern_dist), (10, 10))

    # print((quatern_fault))
    #print(quatern_log)
    #print(quatdistanceMatrix)
"""







"""
from data.Quaternionlog import Quaternion

# Allocate
# quatern_fault = np.zeros((4,(len(samples._data)+1)))
quatern_fault = np.zeros((4, 5))
quatern_log = np.zeros((4, 5))
quatern_dist = np.zeros((4, 5))
# my_quaternion = Quaternion(quatern_fault)

# quatern_log_prev = np.zeros((4, 23))
quatern_fault[:, 0] = [1., 0., 0., 0.]
# h is the time step, delta_t, which is constant for the momoent as 0.02s (50Hz of data acquisition), but you can also make it variable too
h = 0.02

# for i in range(1,len(samples._data)-1):
for i in range(1, 5):
    # quatern_fault [:,i]= samples.RK4((samples.KinematicModel(samples._data[:i][i-1][1:4], quatern_fault[:,0])), samples._data[:i][i-1][1:4], quatern_fault[:,i], h)
    function = samples.KinematicModel(samples._data[:i][i - 1][1:4], quatern_fault[:, 0])
    quatern_fault[:, i] = samples.RK4(samples.KinematicModel, np.array(samples._data[:i][i - 1][1:4]),
                                      quatern_fault[:, i - 1], h)
    # prev_quaternion = Quaternion(quatern_fault[:,i-1])
    my_quaternion = Quaternion(np.array(quatern_fault[:, i]))
    # quatern_log_prev[:, i-1] = Quaternion.log(prev_quaternion)
    # print(my_quaternion)
    quatern_log[:, i] = Quaternion.log(Quaternion(np.array(quatern_fault[:, i])))
    # quatern_log[:, i] = Quaternion.log(Quaternion((quatern_fault[:,i])))
    #quatern_dist[:, i] = Quaternion.distance(Quaternion((quatern_fault[:, i])), Quaternion((quatern_fault[:, i - 1])))

print((quatern_fault))
print(quatern_log)
"""
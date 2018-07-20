import util.filereader as rd
#from scipy.integrate import quad
import math
import numpy as np
from data.cluster import Clustering

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")
    #print(samples._data)
    #print(samples[33.38:330.513])
    #print(samples[33.447])
    #print(range(len(list(samples))))
    #print(samples.distance())
    #print(samples.integrand(samples._times))

#integration with quad (But it didnot work)
    #finaldistances=[]
    #for time in range(50):
        #print(samples.integrand(samples._times[0]))
        #ans, err = quad(samples.integrand(samples._times), samples._times[time], samples._times[time+10])
        #finaldistances.append(ans)
        #print(finaldistances)

#Integration of the square of L2norm for all the distances
    #from sympy import *
    #finaldistances=[]
    #for time in range(5):
        #for m in range(1, 6):
            #for n in range(1,6):
                #for k in range(6):
                #print(samples._data[:m][m-1][0])
                #print(samples._data[:(n)][n-1][0])
                #dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(n + 1)][n])])
                #dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:n][n-1])])
            #t = Symbol('t')
            #ans = integrate(dist, (t , samples._times[time], samples._times[time+5]))
            #sol = math.sqrt(ans/(samples._times[time+5]-samples._times[time]))
            #finaldistances.append(sol)
    #print(np.reshape(np.asarray(finaldistances),(5,5)))


# to compute to integration of the square of L2norm for all the trajectories
    from sympy import *
    finaldistances = []
    for m in range(1, 101):
        for n in range(1, 101):
            # dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(n + 1)][n])])
            dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:n][n - 1])])
            t = Symbol('t')
            ans = integrate(dist, (t, samples._data[:m][m-1][0], samples._data[:(m+5)][m][0]))
            sol = math.sqrt(ans / (samples._data[:(m+5)][m][0] - samples._data[:m][m-1][0]))
            finaldistances.append(sol)
    distanceMatrix = np.reshape(np.asarray(finaldistances), (100, 100))
    print(distanceMatrix)



#print(Clustering(4,distanceMatrix))





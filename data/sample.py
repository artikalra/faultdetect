import bisect
import math
from scipy.integrate import quad

# Sample Class
class Samples:

    def __init__(self):
        self._times = []
        self._data = []

    def insert(self, data):
        self._times.append(data[0])
        self._data.append(data)

    @property
    def size(self):
        return len(self._times)

    @property
    def times(self):
        return self._times

    @property
    def samples(self):
        return self._data

    def __getitem__(self, t):
        if isinstance(t, slice):
            low = bisect.bisect_left(self._times, t.start)
            high = bisect.bisect_left(self._times, t.stop)
            return self._data[low:high]
        else:
            n = bisect.bisect_left(self._times, t)
            return self._data[n]


# Things to fix:
# It is to be checked that time between any two samples is constant or not
# also a and b represent ax, ay, az but here it is taking all parameters in the data

#This function is just to calculate the L2Distances
    def distance(self):
        L2distances = []
        #for m in range(1,(len(self._data)-1)):
        for m in range(1,50 ):
            #for k in range(6):
                dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self._data[:m][m-1], self._data[:(m+1)][m])]))
                #dist = math.sqrt(sum((self._data[:(m+1)][0][k] - self._data[:(m+2)][0][k])  ))
                print("Euclidean distance between the accelerometer and gyro parameters: ", dist)
                L2distances.append(dist)
        return L2distances



# This integration should be clubbed with the L2distance calculated above
#def integrand(x):
#    return x ** 2
#ans, err = quad(integrand, 0, 1)
#print(ans)

#This function is again the to find the L2distances to give as an input for the integration
    def integrand(self,t):
        for time in range(len(self._times)):
            l2distances = []
                #for m in range(1, len(list(self)) - 1):
            for m in range(1, 41):
                        # for k in range(6):
                dist = sum([(a - b) ** 2 for a, b in zip(self._data[:m][m-1], self._data[:m + 1][m])])
                #print("Euclidean distance between the accelerometer and gyro parameters: ", dist)
                l2distances.append(dist)
            return l2distances

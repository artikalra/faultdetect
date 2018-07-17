import bisect
import math
from scipy.integrate import quad


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
#
    def distance(self):
        L2distances = []
        #l = list(self)
        for m in range(1,len(list(self))-1):
            #for k in range(6):
                dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self._data[:m][0], self._data[:m+1][0])]))
                print("Euclidean distance between the accelerometer and gyro parameters: ", dist)
                L2distances.append(dist)
        return L2distances



# This integration should be clubbed with the L2distance calculated above
#def integrand(x):
#    return x ** 2
#ans, err = quad(integrand, 0, 1)
#print(ans)

#Things to fix
# There are some issues in the integration part.
# The limits of the integral are not given properly that is why the answer is wrong
    def integrand(self):
        FinalDistances =[]
        for time in range(len(self._times)):
            def distance(self,t):
                L2distances = []
                # l = list(self)
                for m in range(1, len(list(self)) - 1):
                    # for k in range(6):
                    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self._data[:m][0], self._data[:m + 1][0])]))
                    print("Euclidean distance between the accelerometer and gyro parameters: ", dist)
                    L2distances.append(dist)
                    return L2distances
                    return dist
                    ans, err = quad(integrand, self._times[time], self._times[time+1])
                    FinalDistances.append(ans)
                    return FinalDistances

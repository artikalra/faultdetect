import bisect
import math
from scipy.integrate import quad
import numpy as np
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


#to obtain quaternions
    def KinematicModel(self,w,q):
        # inputs : state_prev  .:. states from the previous time t - 1.
        #                          q = [q0 q1 q2 q3 p q r]'
        #                          where;
        #                               q = q0 + q1 * i + q2 * j + q3 * k
        #                               w .:. describes the angular motion of the body
        #                                     frame b with respect to navigation frame
        #                                     North East Down(NED), expressed in body frame
        #                               w = [p q r]'
        # outputs : q_dot  .:. time derivative of quaternions
        #
        # q .:. quaternion
        # q = q0 + q1 * i + q2 * j + q3 * k
        q0,q1,q2,q3 = q
        # w .:. angular velocity vector with components p, q, r
        # w = [p q r]'
        # w describes the angular motion of the body frame b with respect to
        # navigation frame NED, expressed in body frame.
        quat_normalize_gain = 1.0
        m1 = np.array([[-q1,-q2,-q3],[q0,-q3,q2],[q3,q0,-q1],[-q2,q1,q0]])
        omega= np.array(w)
        q_dot = 1.0/2.0*m1.dot(omega) + quat_normalize_gain*(1-(q0**2 + q1**2 + q2**2 + q3**2))*q
        return q_dot


    #to integrate using Runge-kutta
    def RK4(self, f, w0, q0, h):
        w,q = w0,q0
        k1 = h * f(w, q)
        k2 = h * f(w + 0.5 * h, q + 0.5 * k1)
        k3 = h * f(w + 0.5 * h, q + 0.5 * k2)
        k4 = h * f(w + h, q + k3)
        q = q + (k1 + k2 + k2 + k3 + k3 + k4) / 6. # *h is missing or not ?
        return q

import util.filereader as rd
from scipy.integrate import quad

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

#Integration of the square of L2norm
    from sympy import *
    finaldistances=[]
    for time in range(50):
        for m in range(1, 51):
                # for k in range(6):
            dist = sum([(a - b) ** 2 for a, b in zip(samples._data[:m][m - 1], samples._data[:(m + 1)][m])])
            t = Symbol('t')
        ans = integrate(dist, (t , samples._times[time], samples._times[time+1]))
        sol = ans/(samples._times[time+1]-samples._times[time])
        finaldistances.append(sol)
    print(finaldistances)








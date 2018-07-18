import util.filereader as rd
from scipy.integrate import quad

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")
   #print(samples._data)
   # print(samples[33.38:330.513])
   # print(samples[33.447])
   #print(range(len(list(samples))))
   #print(samples.distance())
    print(samples.integrand())
    #finaldistances=[]
    #for time in range(50):
      #  ans = quad(samples.integrand, samples._times[time], samples._times[time+1])
      #  finaldistances.append(ans)
       # print(finaldistances)









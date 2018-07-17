import util.filereader as rd

if __name__ == '__main__':
    samples = rd.parse("data/17_07_06__10_21_07_SD.data")
   # print(samples._data[:1][0][0])
   # print(samples[33.38:330.513])
   # print(samples[33.447])
    print(range(len(list(samples))))
    print(samples.distance())



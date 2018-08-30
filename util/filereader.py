from data.sample import Samples
#from time import sleep

def parse(filename):
    samples = Samples()
    with open(filename, "r") as f:
        p = 0
        # default values for faults as a start, and we will update them whenever we received a setting change...
        sj = float(1.0)
        sk = float(1.0)
        sl = float(0.0) # correct these default fault values !
        sm = float(0.0)
        pm = int(0) # PPRZ_MODE
        al = float(185.0)
        for line in f:
            fields = line.split()
            #print('fields : ', fields)
            t = float(fields[0])
            id = int(fields[1])
            if fields[2] == "IMU_GYRO":
                ox = float(fields[3])
                oy = float(fields[4])
                oz = float(fields[5])
                p += 1
            elif fields[2] == "IMU_ACCEL":
                ax = float(fields[3])
                ay = float(fields[4])
                az = float(fields[5])
                p += 1
            elif fields[2] == "PPRZ_MODE":
                pm = float(fields[3])
            elif fields[2] == "GPS":
                al = float(fields[7])/1000.
            elif fields[2] == "SETTINGS":
                #print('We have a settings !',float(fields[3]),float(fields[4]),float(fields[5]),float(fields[6]) )
                #sleep(15.)
                sj = float(fields[3])
                sk = float(fields[4])
                sl = float(fields[5])
                sm = float(fields[6])
                #p += 1
            if p == 2:
                samples.insert((t, ox, oy, oz, ax, ay, az, sj, sk, sl, sm, pm, al))
                p = 0
    return samples
#
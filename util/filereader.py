from data.sample import Samples

def parse(filename):
    samples = Samples()
    with open(filename, "r") as f:
        p = 0
        for line in f:
            fields = line.split()
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
            if p == 2:
                samples.insert((t, ox, oy, oz, ax, ay, az))
                p = 0
    return samples

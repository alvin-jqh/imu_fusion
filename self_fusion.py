import imufusion
import matplotlib.pyplot as plt
import numpy

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = numpy.genfromtxt("data_files/forward_back.csv", delimiter=",", skip_header=1)

sample_rate = 50 

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

figure, axes = plt.subplots(nrows=3, sharex=True)

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyroscope X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyroscope Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
axes[0].set_ylabel("Degrees/s")
axes[0].set_title("Gyroscope Readings")
axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Accelerometer X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Accelerometer Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
axes[1].set_ylabel("g")
axes[1].set_title("Accelerometer Readings")
axes[1].grid()
axes[1].legend()

axes[2].plot(timestamp, magnetometer[:, 1], "tab:green", label="Magnetometer Y")
axes[2].plot(timestamp, magnetometer[:, 0], "tab:red", label="Magnetometer X")
axes[2].plot(timestamp, magnetometer[:, 2], "tab:blue", label="Magnetometer Z")
axes[2].set_ylabel("uF")
axes[2].set_title("Magnetometer Readings")
axes[2].grid()
axes[2].legend()

axes[2].set_xlabel("time/s")

plt.show()
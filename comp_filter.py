import ahrs
from ahrs import Quaternion
import matplotlib.pyplot as plt
import numpy as np

filename = f"roll_rotation.csv"
path = f"data_files/{filename}"

# import data from csv file
# gyro is in dps, accel in g, mag is in uT
data = np.genfromtxt(path, delimiter=",", skip_header=1)

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

# soft iron correction
A = np.array(
    [[1.0166, -0.0141, -0.0046],
    [-0.0141, 0.9878, 0.0090],
    [-0.0046, 0.0090, 0.9961]]      
)

# hard iron correction
b = np.array([-1.7561, -0.7667, -9.914])

calibrated_mag = magnetometer - b
calibrated_mag = np.dot(calibrated_mag, A)
calibrated_mag[:,1:3] = -calibrated_mag[:,1:3]
# calibrated_mag = magnetometer

sample_freq = 50        # it hertz
gain = 0.98             # gyro to accel/mag

comp = ahrs.filters.complementary.Complementary(gyr=gyroscope,
                                                acc=accelerometer,
                                                mag=calibrated_mag,
                                                frequency=sample_freq,
                                                gain=gain)

euler = np.empty((len(timestamp),3))
Qs = comp.Q

for index in range(len(timestamp)):
    Q = Quaternion(q=np.transpose(Qs[index]))
    euler[index] = Q.to_angles()

euler = euler * 180/np.pi
# Plotting
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

axes[2].plot(timestamp, calibrated_mag[:, 1], "tab:green", label="Magnetometer Y")
axes[2].plot(timestamp, calibrated_mag[:, 0], "tab:red", label="Magnetometer X")
axes[2].plot(timestamp, calibrated_mag[:, 2], "tab:blue", label="Magnetometer Z")
axes[2].set_ylabel("uT")
axes[2].set_title("Magnetometer Readings")
axes[2].grid()
axes[2].legend()

axes[2].set_xlabel("time/s")

plt.figure()

plt.plot(timestamp, euler[:, 0], "tab:red", label="Roll")
plt.plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
plt.plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
plt.ylabel("Degrees")
plt.xlabel("Time/s")
plt.grid()
# plt.ylim((-200,200))
plt.legend()

plt.show()
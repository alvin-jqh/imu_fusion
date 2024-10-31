import imufusion
import matplotlib.pyplot as plt
import numpy as np

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = np.genfromtxt("data_files/y_accel.csv", delimiter=",", skip_header=1)

# sample rate of our sensor
sample_rate = 50 

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
# calibrated_mag = magnetometer
calibrated_mag[:,1:3] = -calibrated_mag[:,1:3]

figure, axes = plt.subplots(nrows=5, sharex=True)

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
axes[2].set_ylabel("uF")
axes[2].set_title("Magnetometer Readings")
axes[2].grid()
axes[2].legend()

axes[4].set_xlabel("time/s")

# initialise the AHRS algo
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_ENU,
                                   0.5,  # gain, chooses between gyro or mag+accel
                                   250,  # gyroscope range 250 for icm
                                   10,  # acceleration rejection
                                   10,  # magnetic rejection
                                   5 * sample_rate)  # rejection timeout = 5 seconds

# Process sensor data
delta_time = np.diff(timestamp, prepend=timestamp[0])

euler = np.empty((len(timestamp), 3))
internal_states = np.empty((len(timestamp), 6))
flags = np.empty((len(timestamp), 4))
acceleration = np.empty((len(timestamp), 3))
gravity = np.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update(gyroscope[index], accelerometer[index], calibrated_mag[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                       ahrs_internal_states.accelerometer_ignored,
                                       ahrs_internal_states.acceleration_recovery_trigger,
                                       ahrs_internal_states.magnetic_error,
                                       ahrs_internal_states.magnetometer_ignored,
                                       ahrs_internal_states.magnetic_recovery_trigger,])

    acceleration[index] = ahrs.linear_acceleration  
    gravity[index] = ahrs.gravity

    ahrs_flags = ahrs.flags
    flags[index] = np.array(
        [   ahrs_flags.initialising,
            ahrs_flags.angular_rate_recovery,
            ahrs_flags.acceleration_recovery,
            ahrs_flags.magnetic_recovery,
        ])
    
# Identify moving periods
is_moving = np.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 5/9.81

margin = int(0.1 * sample_rate)  # 100 ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

# Calculate velocity (includes integral drift)
velocity = np.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# Calculate position
position = np.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

## bunch of plotting    
axes[3].plot(timestamp, gravity[:, 0], "tab:red", label="Gravity X")
axes[3].plot(timestamp, gravity[:, 1], "tab:green", label="Gravity Y")
axes[3].plot(timestamp, gravity[:, 2], "tab:blue", label="Gravity Z")
axes[3].set_ylabel("g")
axes[3].set_title("Gravity Readings")
axes[3].grid()
axes[3].legend()

axes[4].plot(timestamp, acceleration[:, 0], "tab:red", label="Acceleration X")
axes[4].plot(timestamp, acceleration[:, 1], "tab:green", label="Acceleration Y")
axes[4].plot(timestamp, acceleration[:, 2], "tab:blue", label="Acceleration Z")
axes[4].set_ylabel("g")
axes[4].set_title("Linear Acceleration")
# axes[4].set_ylim((-3, 3))
axes[4].grid()
axes[4].legend()
    
def plot_bool(axis, x, y, label):
    axis.plot(x, y, "tab:cyan", label=label)
    plt.sca(axis)
    plt.yticks([0, 1], ["False", "True"])
    axis.grid()
    axis.legend()

# Plot Euler angles
figure2, axes2 = plt.subplots(nrows=11, sharex=True, gridspec_kw={"height_ratios": [6, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

figure2.suptitle("Euler angles, internal states, and flags")

axes2[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes2[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes2[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes2[0].set_ylabel("Degrees")
axes2[0].grid()
axes2[0].set_ylim((-200,200))
axes2[0].legend()

# Plot initialising flag
plot_bool(axes2[1], timestamp, flags[:, 0], "Initialising")

# Plot angular rate recovery flag
plot_bool(axes2[2], timestamp, flags[:, 1], "Angular rate recovery")

# Plot acceleration rejection internal states and flag
axes2[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes2[3].set_ylabel("Degrees")
axes2[3].grid()
axes2[3].legend()

plot_bool(axes2[4], timestamp, internal_states[:, 1], "Accelerometer ignored")

axes2[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes2[5].grid()
axes2[5].legend()

plot_bool(axes2[6], timestamp, flags[:, 2], "Acceleration recovery")

# Plot magnetic rejection internal states and flag
axes2[7].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
axes2[7].set_ylabel("Degrees")
axes2[7].grid()
axes2[7].legend()

plot_bool(axes2[8], timestamp, internal_states[:, 4], "Magnetometer ignored")

axes2[9].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
axes2[9].grid()
axes2[9].legend()

plot_bool(axes2[10], timestamp, flags[:, 3], "Magnetic recovery")

fig3 = plt.figure()
axes3 = fig3.add_subplot(projection='3d')
axes3.scatter(position[:,0], position[:,1], timestamp)

axes3.set_xlabel('X')
axes3.set_ylabel('Y')
axes3.set_zlabel('Time')

plt.show()
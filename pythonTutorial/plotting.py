import numpy as np
import matplotlib.pyplot as plt

x_axis = np.arange(0,20,1)
y_axis = np.arange(0,11,1)
x_values = x_axis

def displacement_with_acceleration(initialDiplacement, initialVelocity, acceleration, time):
    return initialDiplacement + initialVelocity * time + acceleration * time**2 / 2

y_values = []

for x_value in x_values:
    y_values.append(displacement_with_acceleration(0, 100, -9.8, x_value))

plt.scatter(x_values, y_values, s=x_values * 10, marker="s")
plt.show()

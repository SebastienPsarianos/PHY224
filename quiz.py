import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = np.array([0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80])
y = np.array(
    [-3.46, -4.78, -6.32, -6.20, -8.60, -11.57, -12.27, -15.70, -16.71]
)


def quadratic_fit(x, A, B, C):
    return A * x ** 2 + B * x + C


data, covariance = curve_fit(quadratic_fit, x, y)
print(data[0])

plt.scatter(x, y)
plt.plot(x, quadratic_fit(x, data[0], data[1], data[2]))
plt.draw()
plt.show()

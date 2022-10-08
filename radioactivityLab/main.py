import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson


# Part 1)

# Defining the model functions.
def linear_model(values, a, b) -> any:
    return a * values + b


def exponential_model(values, a, b) -> any:
    return b * np.exp(a * values)


def theoretical_model(values, b) -> any:
    return b * np.exp((-1 / 156 * np.log(2)) * values)


# Reading the data into Python.
_, total = np.loadtxt(
    "Barium137_20220929AM_20min20sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True)
_, background = np.loadtxt(
    "Background20220929AM-20min20s.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True)

# Calculating the mean background radiation and subtracting it from the data.
mean_background = np.average(background)
sample_count = total - mean_background

# Removing all values less than or equal to 0 to avoid mathematical
# issues with the logarithmic co-domain for the linear regression
lst = []
lst2 = []
lst3 = []
for i in range(len(sample_count)):
    if sample_count[i] > 0:
        lst.append(sample_count[i])
        lst2.append(total[i])
        lst3.append(background[i])

new_sample_count = np.array(lst)
new_total = np.array(lst2)
new_background = np.array(lst3)


# Calculating the uncertainty for each point.
def error_propagation(uncertainty_one, uncertainty_two) -> any:
    """Return the uncertainty of the sample.
    """
    return np.sqrt(uncertainty_one + uncertainty_two)


sample_count_uncertainty = error_propagation(new_total, new_background)


# Convert the count data into rates.
def count_rate(events, sample_time) -> tuple:
    """Return the count rate and its uncertainty.
    """
    return events / sample_time, np.sqrt(events) / sample_time


sample_rate, sample_rate_uncertainty = count_rate(new_sample_count, 20)

# Finding the theoretical coefficient for the theoretical curve.
time = np.arange(20, 1181, 20)
popt3, pcov3 = curve_fit(theoretical_model, time, new_sample_count)

# Performing the linear regression on the data using the linear model.
popt1, pcov1 = curve_fit(linear_model, time, np.log(new_sample_count),
                         p0=(-1 / 156 * np.log(2), np.log(popt3[0])))
pvar1 = np.sqrt(np.diag(pcov1))
uncertainty1 = pvar1[0] / popt1[0] ** 2
print("The half-life for the linear regression method is " +
      str(-1 / popt1[0] * np.log(2)) + " ± " + str(uncertainty1) + "s.")
print("The intensity at t = 0 for the linear model is " + str(np.exp(popt1[1]))
      + " ± " + str(pvar1[1] * np.exp(popt1[1])))

# Performing the nonlinear regression on the data using the nonlinear model.
popt2, pcov2 = curve_fit(exponential_model, time, new_sample_count,
                         p0=(-1 / 156 * np.log(2), popt3[0]))
pvar2 = np.sqrt(np.diag(pcov2))
uncertainty2 = pvar2[0] / popt2[0] ** 2
print("The half-life for the nonlinear regression method is " +
      str(-1 / popt2[0] * np.log(2)) + " ± " + str(uncertainty2) + "s.")
print("The intensity at t = 0 for the nonlinear model is " + str(popt2[1]) +
      " ± " + str(pvar2[1]))

# Plotting the raw data, models and theoretical curve.
plt.figure(figsize=(8, 6))
plt.plot(time, exponential_model(time, popt1[0], np.exp(popt1[1])),
         label="Linear Model")
plt.plot(time, exponential_model(time, popt2[0], popt2[1]),
         label="Exponential Model")
plt.plot(time, theoretical_model(time, popt3[0]), label="Theoretical Curve")
plt.errorbar(time, new_sample_count, yerr=sample_count_uncertainty, marker="o",
             ls="", lw=1, label="Experimental Data")
plt.scatter(time, exponential_model(time, popt1[0], np.exp(popt1[1])) -
            new_sample_count, color='pink', label="Linear Model Residuals")
plt.scatter(time, exponential_model(time, popt2[0], popt2[1]) -
            new_sample_count, color='purple', label="Nonlinear Model Residuals")
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.legend()
plt.show()

# Plotting using a logarithmic y-axis.
plt.figure(figsize=(8, 6))
plt.plot(time, exponential_model(time, popt1[0], np.exp(popt1[1])),
         label="Linear Model")
plt.plot(time, exponential_model(time, popt2[0], popt2[1]),
         label="Exponential Model")
plt.plot(time, theoretical_model(time, popt3[0]), label="Theoretical Curve")
plt.errorbar(time, new_sample_count, yerr=(sample_count_uncertainty /
                                           new_sample_count), marker="o",
             ls="",
             lw=1, label="Experimental Data")
plt.scatter(time, linear_model(time, popt1[0], (popt1[1])) -
            np.log(new_sample_count), color='pink',
            label="Linear Model Residuals")
plt.scatter(time, linear_model(time, popt2[0], np.log(popt2[1])) -
            np.log(new_sample_count), color='purple',
            label="Nonlinear Model Residuals")
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.yscale("log")
plt.legend()
plt.show()


# Calculating the reduced chi-squared value.
def characterize(y: any, func: any, u: any) -> float:
    """Return the reduced chi-squared metric to determine how well a model
    function fits a given set of data using the measured data <y>, the
    prediction with the model <func> and the uncertainty on each measurement's
    dependent data <u>.
    """
    value = 0

    for i in range(np.size(y)):
        value += ((y[i] - func[i]) ** 2) / (u[i] ** 2)
        i += 1

    return value / (np.size(y) - 2)


print("The reduced chi-squared value for the linear model is " +
      str(characterize(np.log(new_sample_count), linear_model(time, popt1[0],
                                                              popt1[1]),
                       sample_count_uncertainty / new_sample_count)) + ".")
print("The reduced chi-squared value for the nonlinear model is " +
      str(characterize(new_sample_count, exponential_model(time, popt2[0],
                                                           popt2[1]),
                       sample_count_uncertainty)) + ".")

# Part 2)

# Reading the data into Python.
_, total_fiesta = np.loadtxt(
    "FiestaPlate(U)-20220929_20min3sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True)

sample_fiesta = total_fiesta - mean_background

# Plotting the histogram, Poisson and Gaussian Distribution.

sorted_fiesta = np.sort(sample_fiesta)
average_fiesta = np.mean(sorted_fiesta)
x_axis = np.arange(15, 65, 0.5)
y_axis = poisson.pmf(x_axis, mu=average_fiesta)
fit = norm.pdf(sorted_fiesta, average_fiesta, np.sqrt(average_fiesta))

plt.hist(sorted_fiesta, density=True)
plt.plot(x_axis, y_axis, label="Poisson Distribution")
plt.plot(sorted_fiesta, fit, label="Gaussian Distribution")
plt.legend()
plt.show()

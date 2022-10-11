import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson
import os

directory = os.path.dirname(os.path.realpath(__file__))

# ###########
# # Functions
# ###########
# Models.
def linear_model(values, a, b) -> any:
    return a * values + b


def exponential_model(values, a, b) -> any:
    return b * np.exp(a * values)


def theoretical_model(values, b) -> any:
    return b * np.exp((-1 / 156 * np.log(2)) * values)


# Analysis
def logarithmic_error_propagation(value: any, uncertainty: any) -> float:
    """Return the propagated error for the logarithm of a value"""
    return abs(uncertainty / value)


def calculate_uncertainty(count, mean_background) -> any:
    """Return the uncertainty of the sample.
    """
    return np.sqrt(count + mean_background)


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


def count_rate(events, sample_time) -> tuple:
    """Return the count rate and its uncertainty.
    """
    return events / sample_time, np.sqrt(events) / sample_time


########
# Part 1
########
# Reading the data into Python.
_, total = np.loadtxt(
    f"{directory}/Barium137_20220929AM_20min20sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)
_, background = np.loadtxt(
    f"{directory}/Background20220929AM-20min20s.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

# # Calculating the mean background radiation and subtracting it from the data.
mean_background = np.average(background)
sample_count = total - mean_background

# # Setting up time intervals
time = np.arange(20, 1201, 20)

# Removing all values less than or equal to 0 to avoid mathematical
# issues with the logarithmic co-domain for the linear regression
zeroOrLess = [i for i in range(len(sample_count)) if sample_count[i] <= 0]
time = np.delete(time, zeroOrLess)
sample_count = np.delete(sample_count, zeroOrLess)
total = np.delete(total, zeroOrLess)
background = np.delete(background, zeroOrLess)


# Calculating the uncertainty for each point.
sample_count_uncertainty = calculate_uncertainty(total, mean_background)

# Convert the count data into rates.
sample_rate, sample_rate_uncertainty = count_rate(sample_count, 20)

# Finding the theoretical coefficient for the theoretical curve.
popt3, pcov3 = curve_fit(theoretical_model, time, sample_count)

# Performing the linear regression on the data using the linear model.
popt1, pcov1 = curve_fit(
    linear_model,
    time,
    np.log(sample_count),
    p0=(-1 / 156 * np.log(2), np.log(popt3[0])),
)
pvar1 = np.sqrt(np.diag(pcov1))
uncertainty1 = pvar1[0] / popt1[0] ** 2
print(
    "The half-life for the linear regression method is "
    + str(-1 / popt1[0] * np.log(2))
    + " ± "
    + str(uncertainty1)
    + "s."
)
print(
    "The intensity at t = 0 for the linear model is "
    + str(np.exp(popt1[1]))
    + " ± "
    + str(pvar1[1] * np.exp(popt1[1]))
)

# Performing the exponential regression on the data using the exponential model.
popt2, pcov2 = curve_fit(
    exponential_model, time, sample_count, p0=(-1 / 156 * np.log(2), popt3[0]),
)
pvar2 = np.sqrt(np.diag(pcov2))
uncertainty2 = pvar2[0] / popt2[0] ** 2
print(
    "The half-life for the exponential regression method is "
    + str(-1 / popt2[0] * np.log(2))
    + " ± "
    + str(uncertainty2)
    + "s."
)
print(
    "The intensity at t = 0 for the exponential model is "
    + str(popt2[1])
    + " ± "
    + str(pvar2[1])
)

# Plotting the raw data, models and theoretical curve.
plt.figure(figsize=(8, 6))
plt.plot(
    time, np.exp(linear_model(time, popt1[0], popt1[1])), label="Linear Model",
)
plt.plot(
    time, exponential_model(time, popt2[0], popt2[1]), label="Exponential Model"
)
plt.plot(time, theoretical_model(time, popt3[0]), label="Theoretical Curve")
plt.errorbar(
    time,
    sample_count,
    yerr=sample_count_uncertainty,
    marker="o",
    ls="",
    lw=1,
    label="Experimental Data",
)
plt.scatter(
    time,
    exponential_model(time, popt1[0], np.exp(popt1[1])) - sample_count,
    color="pink",
    label="Linear Model Residuals",
)
plt.scatter(
    time,
    exponential_model(time, popt2[0], popt2[1]) - sample_count,
    color="purple",
    label="Exponential Model Residuals",
)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.legend()
plt.savefig(f"{directory}/figures/linearBariumGraph.pdf")
plt.cla()

# Plotting using a logarithmic y-axis.
plt.figure(figsize=(8, 6))
plt.plot(
    time, linear_model(time, *popt1), label="Linear Model",
)

plt.plot(
    time,
    np.log(exponential_model(time, popt2[0], popt2[1])),
    label="Exponential Model",
)

plt.plot(
    time, np.log(theoretical_model(time, popt3[0])), label="Theoretical Curve"
)

plt.errorbar(
    time,
    np.log(sample_count),
    yerr=logarithmic_error_propagation(sample_count, sample_count_uncertainty),
    marker="o",
    ls="",
    lw=1,
    label="Experimental Data",
)

plt.scatter(
    time,
    linear_model(time, popt1[0], (popt1[1])) - np.log(sample_count),
    color="pink",
    label="Linear Model Residuals",
)
plt.scatter(
    time,
    linear_model(time, popt2[0], np.log(popt2[1])) - np.log(sample_count),
    color="purple",
    label="Exponential Model Residuals",
)
plt.xlabel("Time (s)")
plt.ylabel("Sample Count")
plt.legend()
plt.savefig(f"{directory}/figures/logBariumGraph.pdf")

plt.cla()


# Calculating the reduced chi-squared value.
print(
    "The reduced chi-squared value for the linear model is "
    + str(
        characterize(
            np.log(sample_count),
            linear_model(time, popt1[0], popt1[1]),
            sample_count_uncertainty / sample_count,
        )
    )
    + "."
)
print(
    "The reduced chi-squared value for the exponential model is "
    + str(
        characterize(
            sample_count,
            exponential_model(time, popt2[0], popt2[1]),
            sample_count_uncertainty,
        )
    )
    + "."
)

# # Part 2)

# # Reading the data into Python.
_, total_fiesta = np.loadtxt(
    f"{directory}/FiestaPlate(U)-20220929_20min3sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

sample_fiesta = total_fiesta - mean_background

# Plotting the histogram, Poisson and Gaussian Distribution for the Fiesta plate
# and background data.

sorted_fiesta = np.sort(sample_fiesta)
sorted_background = np.sort(background)
average_fiesta = np.mean(sorted_fiesta)

# X axis for Poisson distribution
x_axis_fiesta_poisson = np.arange(15, 65, 1)
x_axis_background_poisson = np.arange(0, 10, 1)

# X axis for background distribution
x_axis_fiesta_gauss = np.arange(15, 65, 0.01)
x_axis_background_gauss = np.arange(0, 10, 0.01)


plt.hist(sorted_fiesta, density=True)
plt.plot(
    x_axis_fiesta_poisson,
    poisson.pmf(x_axis_fiesta_poisson, mu=average_fiesta),
    label="Poisson Distribution",
)
plt.plot(
    x_axis_fiesta_gauss,
    norm.pdf(x_axis_fiesta_gauss, average_fiesta, np.sqrt(average_fiesta)),
    label="Gaussian Distribution",
)
plt.xlabel("Number of counts over 3s interval")
plt.ylabel("Probability Density")
plt.xlim(0, 80)
plt.legend()
plt.savefig(f"{directory}/figures/fiestaDistributionGraph.pdf")
plt.cla()

plt.hist(sorted_background, density=True)
plt.plot(
    x_axis_background_poisson,
    poisson.pmf(x_axis_background_poisson, mu=mean_background),
    label="Poisson Distribution",
)
plt.plot(
    x_axis_background_gauss,
    norm.pdf(
        x_axis_background_gauss, mean_background, np.sqrt(mean_background)
    ),
    label="Gaussian Distribution",
)
plt.xlabel("Number of counts over 20s interval")
plt.ylabel("Probability Density")
plt.xlim(0, 10)
plt.legend()
plt.savefig(f"{directory}/figures/backgroundDistributionGraph.pdf")
plt.cla()


from functions import *

voltages, currents = np.loadtxt(
    f"{directory}/data/experimentThree.csv", delimiter=",", unpack=True
)

voltageUncertainties = calculate_uncertainty(voltages, 0.001, 0.05, 2)
currentUncertainties = calculate_uncertainty(currents, 0.01, 0.2, 5)

logVoltages = np.log(voltages)
logCurrents = np.log(currents)

popt1, pcov1 = curve_fit(linear_model, logVoltages, logCurrents)
pvar1 = np.sqrt(np.diag(pcov1))

popt2, pcov2 = curve_fit(nonlinear_model, voltages, currents)
pvar2 = np.sqrt(np.diag(pcov2))

# Determining the theoretical constant of proportionality based on the measured
# data.
popt3, pcov3 = curve_fit(model_light, voltages, currents)

print(popt3)

# Plotting the data.

plt.plot(
    voltages,
    nonlinear_model(voltages, popt2[0], popt2[1]),
    marker="",
    color="k",
    label="Curve of Best fit: Nonlinear Model",
)
plt.plot(
    voltages,
    nonlinear_model(voltages, np.exp(popt1[1]), popt1[0]),
    marker="",
    color="b",
    label="Curve of Best fit: Linear Model",
)
plt.plot(
    voltages,
    model_light(4.576, voltages),
    marker="",
    color="pink",
    label="Theoretical Curve",
)
plt.errorbar(
    voltages,
    currents,
    xerr=voltageUncertainties,
    yerr=currentUncertainties,
    capsize=4,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.xlabel("Voltage Across Resistor (V)")
plt.ylabel("Current (mA)")
plt.legend(loc="upper left")
plt.savefig(f"{directory}/graphs/experimentThreeOne.pdf", format="pdf")
plt.show()
plt.cla()

# Data plotted on a log-log plot.

plt.plot(
    voltages,
    nonlinear_model(voltages, popt2[0], popt2[1]),
    marker="",
    color="k",
    label="Curve of Best fit: Nonlinear Model",
)
plt.plot(
    voltages,
    nonlinear_model(voltages, np.exp(popt1[1]), popt1[0]),
    marker="",
    color="b",
    label="Curve of Best fit: Linear Model",
)
plt.plot(
    voltages,
    model_light(4.576, voltages),
    marker="",
    color="pink",
    label="Theoretical Curve",
)
plt.errorbar(
    voltages,
    currents,
    xerr=voltageUncertainties,
    yerr=currentUncertainties,
    capsize=4,
    c="r",
    label="Experimental Data",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.xlabel("Voltage Across Resistor (V)")
plt.ylabel("Current (mA)")
plt.legend(loc="upper left")
plt.yscale("log")
plt.xscale("log")
plt.savefig(f"{directory}/graphs/experimentThreeTwo.pdf", format="pdf")
plt.show()

print("Exponent and coefficient found from linear regression:")
print(str(popt1[0]) + " ± " + str(pvar1[0]))
print(
    str(np.exp(popt1[1]))
    + " ± "
    + str(error_propagation_exponential(np.exp(popt1[1]), pvar1[1]))
)

print("Exponent and coefficient found from nonlinear regression:")
print(str(popt2[1]) + " ± " + str(pvar2[1]))
print(str(popt2[0]) + " ± " + str(pvar2[0]))

print("Reduced chi-squared for linear model:")
print(
    calculatechisquared(
        np.log(voltages),
        np.log(currents),
        currentUncertainties / currents,
        linear_model,
        *popt1,
    )
)

print("Reduced chi-squared for nonlinear model:")
print(
    calculatechisquared(
        voltages, currents, currentUncertainties, nonlinear_model, *popt2
    )
)

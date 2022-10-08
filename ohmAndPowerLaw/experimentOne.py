from functions import *

voltages, currents = np.loadtxt(
    f"{directory}/data/experimentOne.csv", delimiter=",", unpack=True
)

voltageUncertainties = calculate_uncertainty(voltages, 0.001, 0.05, 2)
currentUncertainties = calculate_uncertainty(currents, 0.001, 0.2, 5)


popt, pcov = curve_fit(linear_model, voltages, currents, p0=(1 / 0.47, 0))
pvar = np.sqrt(np.diag(pcov))

plt.plot(
    voltages,
    linear_model(voltages, popt[0], popt[1]),
    color="k",
    label="Line of Best Fit",
)
plt.scatter(
    voltages,
    linear_model(voltages, popt[0], popt[1]) - currents,
    color="green",
    label="Residual Plot",
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
plt.legend()
plt.savefig(f"{directory}/graphs/experimentOne.pdf")
plt.show()
plt.cla()

print(
    "The resistance is "
    + str(1 / popt[0] * 1000)
    + " ± "
    + str((error_propagation_division(1 / popt[0], popt[0], pvar[0])) * 1000)
    + "."
)

print(
    "The reduced chi-squared value is "
    + str(
        calculatechisquared(
            voltages, currents, currentUncertainties, linear_model, *popt
        )
    )
    + "."
)


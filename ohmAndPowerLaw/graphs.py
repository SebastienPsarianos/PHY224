import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

experimentOne = np.loadtxt(
    "./ohmAndPowerLaw/data/experimentOne.csv", delimiter=",", unpack=True
)
experimentTwo = np.loadtxt(
    "./ohmAndPowerLaw/data/experimentTwo.csv", delimiter=",", unpack=True
)
experimentThree = np.loadtxt(
    "./ohmAndPowerLaw/data/experimentThree.csv", delimiter=",", unpack=True
)


def current_model(voltage, resistance):
    return voltage / resistance * 1000


def linear_model(values, a, b):
    return a * values + b


def nonlinear_model(values, a, b):
    return a * values * b


#
# Uncertainty Calculations Based on Documentation
# for U1270 Series Handheld Digital Multimeters
#
def calculate_uncertainty(value, res, percentage, multiplier):
    return value * percentage / 100 + res * multiplier


def generate_graph(
    data, name, xAxis, yAxis, currentUncertaintyData, voltageUncertaintyData
):
    """
        Generates and saves the graph with x/y values defined in <data>.
        Uses the parse_data function to parse the csv data in <data>.
        Uses the <calculate_uncertainty> function to calculate uncertainties for
            all values.
        The file will be saved in PDF format with the filename <name>.
        Uses the model defined in <model> to model a linear fit
            and displays the slope and covariance.
    """
    voltages, currents = data
    voltageUncertainty = calculate_uncertainty(
        voltages,
        voltageUncertaintyData[0],
        voltageUncertaintyData[1],
        voltageUncertaintyData[2],
    )
    currentUncertainty = calculate_uncertainty(
        currents,
        currentUncertaintyData[0],
        currentUncertaintyData[1],
        currentUncertaintyData[2],
    )

    popt, pcov = curve_fit(current_model, xdata=voltages, ydata=currents)

    plt.errorbar(
        voltages,
        currents,
        xerr=voltageUncertainty,
        yerr=currentUncertainty,
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
    plt.plot(
        voltages,
        current_model(voltages, popt[0]),
        c="k",
        label=f"Linear fit:\nLinear Fit Resistance ~ {np.round(popt[0], 2)}\n\
    Covariance ~ {np.format_float_scientific(pcov[0][0], 3)}",
    )

    plt.xlabel(xAxis)
    plt.ylabel(yAxis)

    plt.legend()
    plt.draw()
    plt.savefig(f"./ohmAndPowerLaw/graphs/{name}.pdf", format="pdf")
    plt.cla()


generate_graph(
    experimentOne,
    "experimentOne",
    "Potential Difference across Resistor (V)",
    "Measured Current (mA)",
    (0.001, 0.2, 5),
    (0.001, 0.05, 2),
)
generate_graph(
    experimentTwo,
    "experimentTwo",
    "Potential Difference across Resistor (Volts)",
    "Measured Current (mA)",
    (0.001, 0.2, 5),
    (0.001, 0.05, 2),
)
generate_graph(
    experimentThree,
    "experimentThree",
    "Potential Difference across Resistor (Volts)",
    "Measured Current (mA)",
    (0.01, 0.2, 5),
    (0.001, 0.05, 2),
)

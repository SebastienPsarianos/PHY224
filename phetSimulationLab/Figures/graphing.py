import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


experiment_one = "11.00,1.100;10.00,1.000;9.00,0.900;8.00,0.800;7.00,0.700;6.00\
    ,0.600;5.00,0.500"
experiment_two = "11.00,0.470;10.00,0.420;9.00,0.380;8.00,0.340;7.00,0.300;6.00\
    ,0.250;5.00,0.210"
experiment_three = "11.00,0.430;10.00,0.390;9.00,0.350;8.00,0.310;7.00,0.270;6.00\
    ,0.240;5.00,0.200"


def parse_data(data: str):
    """Parses and sorts two column csv into correct format for pycharm"""
    lines = data.split(";")

    x_data = []
    y_data = []
    for i in range(len(lines)):
        voltage, current = lines[i].split(",")
        y_data.append(float(current))
        x_data.append(float(voltage))

    x_data.reverse()
    y_data.reverse()

    return (x_data, y_data)


def current_model(voltage, resistance):
    return voltage / resistance


#
# Uncertainty Calculations Based on Documentation
# for U1270 Series Handheld Digital Multimeters
#
def voltage_uncertainty(voltage):
    """Formula based on uncertainty for 300V setting"""
    return voltage * 0.0005 + 0.02


def current_uncertainty(current):
    """Formula based on uncertainty for 10A setting"""
    return current * 0.003 + 0.01


def calculate_uncertainty(voltages, currents):
    """
        Returns Tuple with array of voltage uncertainties for the matching value
    """
    voltageUncertainty = []
    currentUncertainty = []
    for i in range(len(voltages)):
        voltageUncertainty.append(voltage_uncertainty(voltages[i]))
        currentUncertainty.append(current_uncertainty(currents[i]))

    return (voltageUncertainty, currentUncertainty)


#
# End Uncertainty
#


def generate_graph(data, name, xAxis, yAxis):
    """
        Generates and saves the graph with x/y values defined in <data>.
        Uses the parse_data function to parse the csv data in <data>.
        Uses the <calculate_uncertainty> function to calculate uncertainties for
            all values.
        The file will be saved in PDF format with the filename <name>.
        Uses the model defined in <model> to model a linear fit
            and displays the slope and covariance.
    """
    voltages, currents = parse_data(data)
    voltageUncertainty, currentUncertainty = calculate_uncertainty(
        voltages, currents
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
        elinewidth=0.5,
        capthick=1,
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
    plt.savefig(f"{name}.pdf", format="pdf")
    plt.cla()


generate_graph(
    experiment_one,
    "./phetSimulationLab/Figures/experimentOne",
    "Potential Difference across Resistor (Volts)",
    "Measured Current (Amps)",
)
generate_graph(
    experiment_two,
    "./phetSimulationLab/Figures/experimentTwo",
    "Potential Difference across Resistor (Volts)",
    "Measured Current (Amps)",
)
generate_graph(
    experiment_three,
    "./phetSimulationLab/Figures/experimentThree",
    "Potential Difference across Battery (Volts)",
    "Measured Current (Amps)",
)


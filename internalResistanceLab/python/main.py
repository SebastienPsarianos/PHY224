import numpy as np
import matplotlib.pyplot as plt
import os

# Defining linear model.
from scipy.optimize import curve_fit

directory = os.path.dirname(os.path.realpath(__file__))


def linear(current, open_voltage, resistance):
    return open_voltage - resistance * current


# Reading CVS files into Python.
voltage_battery_1, current_battery_1 = np.loadtxt(
    f"{directory}/Battery_1.csv", delimiter=",", unpack=True
)
voltage_battery_2, current_battery_2 = np.loadtxt(
    f"{directory}/Battery_2.csv", delimiter=",", unpack=True
)

voltage_ps_6_5_1, current_ps_6_5_1 = np.loadtxt(
    f"{directory}/PowerSupply6.5V1.csv", delimiter=",", unpack=True
)
voltage_ps_6_5_2, current_ps_6_5_2 = np.loadtxt(
    f"{directory}/PowerSupply6.5V2.csv", delimiter=",", unpack=True
)
voltage_ps_10_1, current_ps_10_1 = np.loadtxt(
    f"{directory}/PowerSupply10V1.csv", delimiter=",", unpack=True
)
voltage_ps_10_2, current_ps_10_2 = np.loadtxt(
    f"{directory}/PowerSupply10V2.csv", delimiter=",", unpack=True
)
voltage_ps_15_1, current_ps_15_1 = np.loadtxt(
    f"{directory}/PowerSupply15V1.csv", delimiter=",", unpack=True
)
voltage_ps_15_2, current_ps_15_2 = np.loadtxt(
    f"{directory}/PowerSupply15V2.csv", delimiter=",", unpack=True
)
voltage_ps_20_1, current_ps_20_1 = np.loadtxt(
    f"{directory}/PowerSupply20V1.csv", delimiter=",", unpack=True
)
voltage_ps_20_2, current_ps_20_2 = np.loadtxt(
    f"{directory}/PowerSupply20V2.csv", delimiter=",", unpack=True
)


# Calculating uncertainties.
def calculate_uncertainty(value, res, percentage, multiplier):
    return value * percentage / 100 + res * multiplier


# Battery uncertainty.
uncertainty_voltage_battery_1 = calculate_uncertainty(
    voltage_battery_1, 0.001, 0.05, 2
)
uncertainty_current_battery_1 = calculate_uncertainty(
    current_battery_1, 0.01, 0.2, 5
)

uncertainty_voltage_battery_2 = calculate_uncertainty(
    voltage_battery_2, 0.001, 0.05, 2
)
uncertainty_current_battery_2 = calculate_uncertainty(
    current_battery_2, 0.01, 0.2, 5
)

# Power supply uncertainty.
uncertainty_voltage_ps_6_5_1 = calculate_uncertainty(
    voltage_ps_6_5_1, 0.001, 0.05, 2
)
uncertainty_current_ps_6_5_1 = calculate_uncertainty(
    current_ps_6_5_1, 0.01, 0.2, 5
)
uncertainty_voltage_ps_6_5_2 = calculate_uncertainty(
    voltage_ps_6_5_2, 0.001, 0.05, 2
)
uncertainty_current_ps_6_5_2 = calculate_uncertainty(
    current_ps_6_5_2, 0.01, 0.2, 5
)
uncertainty_voltage_ps_10_1 = calculate_uncertainty(
    voltage_ps_10_1, 0.001, 0.05, 2
)
uncertainty_current_ps_10_1 = calculate_uncertainty(
    current_ps_10_1, 0.01, 0.2, 5
)
uncertainty_voltage_ps_10_2 = calculate_uncertainty(
    voltage_ps_10_2, 0.001, 0.05, 2
)
uncertainty_current_ps_10_2 = calculate_uncertainty(
    current_ps_10_2, 0.01, 0.2, 5
)
uncertainty_voltage_ps_15_1 = calculate_uncertainty(
    voltage_ps_15_1, 0.001, 0.05, 2
)
uncertainty_current_ps_15_1 = calculate_uncertainty(
    current_ps_15_1, 0.01, 0.2, 5
)
uncertainty_voltage_ps_15_2 = calculate_uncertainty(
    voltage_ps_15_2, 0.001, 0.05, 2
)
uncertainty_current_ps_15_2 = calculate_uncertainty(
    current_ps_15_2, 0.01, 0.2, 5
)
uncertainty_voltage_ps_20_1 = calculate_uncertainty(
    voltage_ps_20_1, 0.001, 0.05, 2
)
uncertainty_current_ps_20_1 = calculate_uncertainty(
    current_ps_20_1, 0.01, 0.2, 5
)
uncertainty_voltage_ps_20_2 = calculate_uncertainty(
    voltage_ps_20_2, 0.001, 0.05, 2
)
uncertainty_current_ps_20_2 = calculate_uncertainty(
    current_ps_20_2, 0.01, 0.2, 5
)

# Calculating the output resistance of the battery and power supply.
popt_battery_1, pcov_battery_1 = curve_fit(
    linear, current_battery_1, voltage_battery_1
)
pvar_battery_1 = np.sqrt(np.diag(pcov_battery_1))
print(
    "The output resistance of the battery is "
    + str(-popt_battery_1[1] * 1000)
    + " ± "
    + str(pvar_battery_1[1] * 1000)
    + " for option 1."
)

popt_battery_2, pcov_battery_2 = curve_fit(
    linear, current_battery_2, voltage_battery_2
)
pvar_battery_2 = np.sqrt(np.diag(pcov_battery_2))
print(
    "The output resistance of the battery is "
    + str(-popt_battery_2[1] * 1000)
    + " ± "
    + str(pvar_battery_2[1] * 1000)
    + " for option 2."
)

popt_ps_6_5_1, pcov_ps_6_5_1 = curve_fit(
    linear, current_ps_6_5_1, voltage_ps_6_5_1
)
pvar_ps_6_5_1 = np.sqrt(np.diag(pcov_ps_6_5_1))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_6_5_1[1] * 1000)
    + " ± "
    + str(pvar_ps_6_5_1[1] * 1000)
    + " for option 1 (6.5V)."
)

popt_ps_6_5_2, pcov_ps_6_5_2 = curve_fit(
    linear, current_ps_6_5_2, voltage_ps_6_5_2
)
pvar_ps_6_5_2 = np.sqrt(np.diag(pcov_ps_6_5_2))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_6_5_2[1] * 1000)
    + " ± "
    + str(pvar_ps_6_5_2[1] * 1000)
    + " for option 2 (6.5V)."
)

popt_ps_10_1, pcov_ps_10_1 = curve_fit(linear, current_ps_10_1, voltage_ps_10_1)
pvar_ps_10_1 = np.sqrt(np.diag(pcov_ps_10_1))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_10_1[1] * 1000)
    + " ± "
    + str(pvar_ps_10_1[1] * 1000)
    + " for option 1 (10V)."
)

popt_ps_10_2, pcov_ps_10_2 = curve_fit(linear, current_ps_10_2, voltage_ps_10_2)
pvar_ps_10_2 = np.sqrt(np.diag(pcov_ps_10_2))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_10_2[1] * 1000)
    + " ± "
    + str(pvar_ps_10_2[1] * 1000)
    + " for option 2 (10V)."
)

popt_ps_15_1, pcov_ps_15_1 = curve_fit(linear, current_ps_15_1, voltage_ps_15_1)
pvar_ps_15_1 = np.sqrt(np.diag(pcov_ps_15_1))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_15_1[1] * 1000)
    + " ± "
    + str(pvar_ps_15_1[1] * 1000)
    + " for option 1 (15V)."
)

popt_ps_15_2, pcov_ps_15_2 = curve_fit(linear, current_ps_15_2, voltage_ps_15_2)
pvar_ps_15_2 = np.sqrt(np.diag(pcov_ps_15_2))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_15_2[1] * 1000)
    + " ± "
    + str(pvar_ps_15_2[1] * 1000)
    + " for option 2 (15V)."
)

popt_ps_20_1, pcov_ps_20_1 = curve_fit(linear, current_ps_20_1, voltage_ps_20_1)
pvar_ps_20_1 = np.sqrt(np.diag(pcov_ps_20_1))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_20_1[1] * 1000)
    + " ± "
    + str(pvar_ps_20_1[1] * 1000)
    + " for option 1 (20V)."
)

popt_ps_20_2, pcov_ps_20_2 = curve_fit(linear, current_ps_20_2, voltage_ps_20_2)
pvar_ps_20_2 = np.sqrt(np.diag(pcov_ps_20_2))
print(
    "The output resistance of the power source is "
    + str(-popt_ps_20_2[1] * 1000)
    + " ± "
    + str(pvar_ps_20_2[1] * 1000)
    + " for option 2 (20V)."
)

# Function which generates plots.
def generate_graph(
    xvalues,
    yvalues,
    xuncertainty,
    yuncertainty,
    open_estimate,
    resistance_estimate,
    i,
):

    plt.errorbar(
        xvalues,
        yvalues,
        xerr=xuncertainty,
        yerr=yuncertainty,
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
        xvalues,
        linear(xvalues, open_estimate, resistance_estimate),
        label="Line of Best Fit",
    )
    plt.xlabel("Current (mA)")
    plt.ylabel("Voltage Across Resistor (V)")
    plt.legend()
    plt.savefig(f"{directory}/../figures/graph{i}.pdf")
    plt.cla()


i = 1
# Battery graphs.
generate_graph(
    current_battery_1,
    voltage_battery_1,
    uncertainty_current_battery_1,
    uncertainty_voltage_battery_1,
    popt_battery_1[0],
    popt_battery_1[1],
    i,
)
i += 1

generate_graph(
    current_battery_2,
    voltage_battery_2,
    uncertainty_current_battery_2,
    uncertainty_voltage_battery_2,
    popt_battery_2[0],
    popt_battery_2[1],
    i,
)
i += 1

# Power supply graphs.
generate_graph(
    current_ps_6_5_1,
    voltage_ps_6_5_1,
    uncertainty_current_ps_6_5_1,
    uncertainty_voltage_ps_6_5_1,
    popt_ps_6_5_1[0],
    popt_ps_6_5_1[1],
    i,
)
i += 1
generate_graph(
    current_ps_6_5_2,
    voltage_ps_6_5_2,
    uncertainty_current_ps_6_5_2,
    uncertainty_voltage_ps_6_5_2,
    popt_ps_6_5_2[0],
    popt_ps_6_5_2[1],
    i,
)
i += 1
generate_graph(
    current_ps_10_1,
    voltage_ps_10_1,
    uncertainty_current_ps_10_1,
    uncertainty_voltage_ps_10_1,
    popt_ps_10_1[0],
    popt_ps_10_1[1],
    i,
)
i += 1
generate_graph(
    current_ps_10_2,
    voltage_ps_10_2,
    uncertainty_current_ps_10_2,
    uncertainty_voltage_ps_10_2,
    popt_ps_10_2[0],
    popt_ps_10_2[1],
    i,
)
i += 1
generate_graph(
    current_ps_15_1,
    voltage_ps_15_1,
    uncertainty_current_ps_15_1,
    uncertainty_voltage_ps_15_1,
    popt_ps_15_1[0],
    popt_ps_15_1[1],
    i,
)
i += 1
generate_graph(
    current_ps_15_2,
    voltage_ps_15_2,
    uncertainty_current_ps_15_2,
    uncertainty_voltage_ps_15_2,
    popt_ps_15_2[0],
    popt_ps_15_2[1],
    i,
)
i += 1
generate_graph(
    current_ps_20_1,
    voltage_ps_20_1,
    uncertainty_current_ps_20_1,
    uncertainty_voltage_ps_20_1,
    popt_ps_20_1[0],
    popt_ps_20_1[1],
    i,
)
i += 1
generate_graph(
    current_ps_20_2,
    voltage_ps_20_2,
    uncertainty_current_ps_20_2,
    uncertainty_voltage_ps_20_2,
    popt_ps_20_2[0],
    popt_ps_20_2[1],
    i,
)
i += 1

# Determining the max current for options 1 and 2

y_axis_1 = np.array(
    [popt_ps_6_5_1[1], popt_ps_10_1[1], popt_ps_15_1[1], popt_ps_20_1[1]]
)
uncertainty_y_1 = np.array(
    [pvar_ps_6_5_1[1], pvar_ps_10_1[1], pvar_ps_15_1[1], pvar_ps_20_1[1]]
)
x_axis_1 = np.array(
    [popt_ps_6_5_1[0], popt_ps_10_1[0], popt_ps_15_1[0], popt_ps_20_1[0]]
)
uncertainty_x_1 = np.array(
    [pvar_ps_6_5_1[0], pvar_ps_10_1[0], pvar_ps_15_1[0], pvar_ps_20_1[0]]
)

y_axis_2 = np.array(
    [popt_ps_6_5_2[1], popt_ps_10_2[1], popt_ps_15_2[1], popt_ps_20_2[1]]
)
uncertainty_y_2 = np.array(
    [pvar_ps_6_5_2[1], pvar_ps_10_2[1], pvar_ps_15_2[1], pvar_ps_20_2[1]]
)
x_axis_2 = np.array(
    [popt_ps_6_5_2[0], popt_ps_10_2[0], popt_ps_15_2[0], popt_ps_20_2[0]]
)
uncertainty_x_2 = np.array(
    [pvar_ps_6_5_2[0], pvar_ps_10_2[0], pvar_ps_15_2[0], pvar_ps_20_2[0]]
)


def division_propagation(
    value_1, uncertainty_1, value_2, uncertainty_2, value_3
):
    return value_3 * np.sqrt(
        (uncertainty_1 / value_1) ** 2 + (uncertainty_2 / value_2) ** 2
    )


# Plotting the terminal voltage vs the maximum current

plt.errorbar(
    x_axis_1,
    x_axis_1 / y_axis_1 / 1000,
    xerr=uncertainty_x_1,
    yerr=division_propagation(
        y_axis_1 * 1000,
        uncertainty_y_1 * 1000,
        x_axis_1,
        uncertainty_x_1,
        x_axis_1 / y_axis_1 / 1000,
    ),
    capsize=4,
    c="r",
    label="Experimental Data " "Using the Option 1 " "Setup",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.errorbar(
    x_axis_2,
    x_axis_2 / y_axis_2 / 1000,
    xerr=uncertainty_x_2,
    yerr=division_propagation(
        y_axis_2 * 1000,
        uncertainty_y_2 * 1000,
        x_axis_2,
        uncertainty_x_2,
        x_axis_2 / y_axis_2 / 1000,
    ),
    capsize=4,
    c="b",
    label="Experimental Data " "Using the Option 2 " "Setup",
    marker="o",
    linestyle="",
    ecolor="k",
    elinewidth=0.25,
    capthick=0.5,
    barsabove=True,
    markersize=5,
)
plt.xlabel("Terminal Voltage (V)")
plt.ylabel("Maximum Current (A)")
plt.legend()
plt.savefig(f"{directory}/../figures/maxCurrent.pdf")


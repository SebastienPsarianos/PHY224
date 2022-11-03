import numpy as np
import os

directory = os.path.dirname(os.path.realpath(__file__))


def multimeterUncertainty(value, valuePercentage, range, rangePercentage):
    return (value * valuePercentage + range * rangePercentage) / 100


currentUncertaintyConstants = (0.18, 3, 0.02)
voltageUncertaintyConstants = (0.002, 1000, 0.0006)


def doNumberStuff(string):
    string = str(round(float(string) / 200, 4))
    if len(string) < 5:
        return string + "0"
    return string


def round_to_one_sig_fig(value):
    num, exp = np.format_float_scientific(value, 0).split(".e")
    return int(num) * 10 ** int(exp)


currents, currentDiameters = np.loadtxt(
    f"{directory}/variedCurrent.csv", unpack=True, delimiter=","
)

voltages, voltageDiameters = np.loadtxt(
    f"{directory}/variedVoltage.csv", unpack=True, delimiter=","
)

rulerUncertainty = round_to_one_sig_fig(0.05 * 3 / 200)

voltageTable = "\\begin{tabular}{|p{0.45\\textwidth} | p{0.45\\textwidth} | }\n"
voltageTable += "\\hline\n"
voltageTable += (
    "Electron gun anode voltage (V) & Measured electron path radius (m)\\\\\n"
)
voltageTable += "\\hline\n"
for i in range(len(voltages)):
    voltageTable += f"""{voltages[i]}$\\pm${round_to_one_sig_fig(multimeterUncertainty(voltages[i],*voltageUncertaintyConstants))}&{doNumberStuff(voltageDiameters[i])}$\\pm${rulerUncertainty}\\\\\n"""
voltageTable += "\\hline\n"
voltageTable += "\\end{tabular}\n"


currentTable = "\\begin{tabular}{|p{0.45\\textwidth} | p{0.45\\textwidth} | }\n"
currentTable += "\\hline\n"
currentTable += "Current through Helmholtz coil (A) & Measured electron path radius (m)\\\\\n"
currentTable += "\\hline\n"
for i in range(len(currents)):
    currentTable += f"""{currents[i]}$\\pm${round_to_one_sig_fig(multimeterUncertainty(currents[i],*currentUncertaintyConstants))}&{doNumberStuff(currentDiameters[i])}$\\pm${rulerUncertainty}\\\\\n"""

currentTable += "\\hline\n"
currentTable += "\\end{tabular}\n"

voltageFile = open(f"{directory}/../latex/voltageTable.tex", "w")
currentFile = open(f"{directory}/../latex/currentTable.tex", "w")


voltageFile.write(voltageTable)
currentFile.write(currentTable)

voltageFile.close()
currentFile.close()

print(multimeterUncertainty(currents[0], *currentUncertaintyConstants))


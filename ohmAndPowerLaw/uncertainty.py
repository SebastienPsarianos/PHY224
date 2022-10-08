from functions import *

dataOne = np.loadtxt(
    f"{directory}/data/experimentOne.csv", delimiter=",", unpack=True
)

dataTwo = np.loadtxt(
    f"{directory}/data/experimentTwo.csv", delimiter=",", unpack=True
)

dataThree = np.loadtxt(
    f"{directory}/data/experimentThree.csv", delimiter=",", unpack=True
)


def round_to_one_sig_fig(value):
    num, exp = np.format_float_scientific(value, 0).split(".e")
    return int(num) * 10 ** int(exp)


def formatTable(data, voltmeterValues, ammeterValues, measurement):
    i = 1
    j = 0

    newString = f"Voltage Across Battery ($V$) & Current ($mA$) & Voltage Across {measurement} ($V$)\\\\\n"
    for j in range(len(data[0])):

        newString += f"${i}.000$&${data[1][j]}\\pm{round_to_one_sig_fig(calculate_uncertainty(data[1][j], *ammeterValues))}$&${data[0][j]}\\pm{round_to_one_sig_fig(calculate_uncertainty(data[0][j], *voltmeterValues))}$\\\\\n"
        i += 1

    return newString


print(formatTable(dataOne, (0.001, 0.05, 2), (0.001, 0.2, 5), "Resistor"))
print(formatTable(dataTwo, (0.001, 0.05, 2), (0.001, 0.2, 5), "Potentiometer"))
print(formatTable(dataThree, (0.001, 0.05, 2), (0.01, 0.2, 5), "Light Bulb"))


import numpy as np

experimentOne = """11.00&11.00&1.100\\\\10.00&10.00&1.000\\\\9.00&9.00&0.900\\\\8.00&8.00&0.800\\\\7.00&7.00&0.700\\\\6.00&6.00&0.600\\\\5.00&5.00&0.500\\\\"""
experimentTwo = """11.00&11.00&0.470\\\\10.00&10.00&0.420\\\\9.00&9.00&0.380\\\\8.00&8.00&0.340\\\\7.00&7.00&0.300\\\\6.00&6.00&0.250\\\\5.00&5.00&0.210\\\\"""
experimentThree = """11.00&10.20&0.430\\\\10.00&9.27&0.390\\\\9.00&8.34&0.350\\\\8.00&7.41&0.310\\\\7.00&6.49&0.270\\\\6.00&5.56&0.240\\\\5.00&4.63&0.200\\\\"""


def voltage_uncertainty(voltage):
    """Formula based on uncertainty for 300V setting"""
    return round_to_one_sig_fig(abs(float(voltage) * 0.0005) + 0.02)


def current_uncertainty(current):
    """Formula based on uncertainty for 10A setting"""
    return round_to_one_sig_fig(abs(float(current) * 0.003) + 0.01)


def round_to_one_sig_fig(value):
    num, exp = np.format_float_scientific(value, 0).split(".e")
    return int(num) * 10 ** int(exp)


def formatTable(data):

    rows = data.split("\\\\")
    newString = ""
    for row in rows:
        if row != "":
            batteryVoltage, resistorVoltage, current = row.split("&")

            newString += f"${batteryVoltage}\\pm{voltage_uncertainty(batteryVoltage)}$& ${resistorVoltage}\\pm{voltage_uncertainty(resistorVoltage)}$& ${current}\\pm{current_uncertainty(current)}$\\\\\n"

    return newString


print(formatTable(experimentOne))
print(formatTable(experimentTwo))
print(formatTable(experimentThree))


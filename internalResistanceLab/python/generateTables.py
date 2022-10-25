import numpy as np
from main import calculate_uncertainty
import os

volt = (0.001, 0.05, 2)
amm = (0.01, 0.2, 5)
resistances = (100, 220, 320, 470, 570, 690, 2700, 2800)

directory = os.path.dirname(os.path.realpath(__file__))

tables = []

tables.append(
    (
        np.loadtxt(f"{directory}/Battery_1.csv", delimiter=",", unpack=True),
        "Battery raw data (Setup 1)",
    )
)


tables.append(
    (
        np.loadtxt(f"{directory}/Battery_2.csv", delimiter=",", unpack=True),
        "Battery raw data (Setup 2)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply6.5V1.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 6.5V (Setup 1)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply6.5V2.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 6.5V (Setup 2)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply10V1.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 10V (Setup 1)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply10V2.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 10V (Setup 2)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply15V1.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 15V (Setup 1)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply15V2.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 15V (Setup 2)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply20V1.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 20V (Setup 1)",
    )
)

tables.append(
    (
        np.loadtxt(
            f"{directory}/PowerSupply20V2.csv", delimiter=",", unpack=True
        ),
        "Power supply raw data at 20V (Setup 2)",
    )
)


def sciNot(num):
    val, exp = np.format_float_scientific(num, 0).split(".e")
    return f"{val}\\times10^{{{int(exp)}}}"


j = 1
string = "\\newpage\n{{\large\\textbf{Raw Data}}} \\\\\n\n"

for table, title in tables:
    string += "\\begin{tabular}{| p{0.3\\textwidth} | p{0.3\\textwidth} | p{0.3\\textwidth} |}\n    \\hline\n"
    string += "    Resistance ($\\Omega$) & Voltage ($V$) & Current ($mA$)\\\\\n    \\hline \n"

    for i in range(len(resistances)):
        string += f"    ${resistances[i]}$ & ${table[0][i]} \\pm {sciNot(calculate_uncertainty(table[0][i], *volt))}$ & ${table[1][i]} \\pm {sciNot(calculate_uncertainty(table[1][i], *amm))}$\\\\\n"

    string += "    \\hline \n \\end{tabular}\\\\\n\n"
    string += f"\\begin{{center}}\n    {{\\textbf{{Table {j}: {title}}} }}\n\end{{center}}\n\\vspace{{10pt}}\n"

    if j % 3 == 0:
        string += "\\newpage\n\n"
    j += 1

file = open(f"{directory}/../latex/test.tex", "w")
file.write(string)
file.close()

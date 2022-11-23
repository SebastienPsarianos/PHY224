import numpy as np
import os

directory = os.path.dirname(os.path.realpath(__file__))

experimentOneAData = np.loadtxt(
    f"{directory}/rawData/experimentOne.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

experimentOneBData = np.loadtxt(
    f"{directory}/rawData/experimentTwo.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

experimentTwoAData = np.loadtxt(
    f"{directory}/rawData/experimentThreePolarized.txt",
    delimiter="\t",
    unpack=True,
)

experimentTwoBData = np.loadtxt(
    f"{directory}/rawData/experimentThreeUnpolarized.txt",
    delimiter="\t",
    unpack=True,
)

angleError = 0.01
intensityError = 0.0001


def round_to_one_sig_fig(value):
    num, exp = np.format_float_scientific(value, 0).split(".e")
    return int(num) * 10 ** int(exp)


def convertMeasuredToIncidence(measurement):
    return np.radians(225 - measurement)


def formatTable(angles, intensities, numPerTable):
    i = 0
    tableStart = (
        "\\begin{tabular}{| p{0.21\\textwidth} | p{0.21\\textwidth} |}\n"
    )
    tableStart += f"\\hline\nIncidence Angle (radians) & Measured Intensity (V)\\\\\n\\hline\n"

    newString = tableStart

    if max(angles) > 5:
        angles = convertMeasuredToIncidence(angles)

    for j in range(len(angles)):
        if j % numPerTable == 0 and j != 0:
            if i % 2 == 0:
                newString += "\\hline\n\\end{tabular}\\hfill\n"
            elif i == len(angles) - 1:
                newString += "\\hline\n\\end{tabular}\n\\newpage\n"
            else:
                newString += "\\hline\n\\end{tabular}\\\\\n"

            newString += tableStart
            i += 1
            print(j, len(angles))
        else:
            newString += f"${round(angles[j],2)} \\pm {angleError}$ & ${intensities[j]} \\pm {intensityError}$\\\\\n"

    return newString + "\\hline\n\\end{tabular}"


experimentOneA = open(f"{directory}/../latex/experimentOneA.tex", "w")
experimentOneB = open(f"{directory}/../latex/experimentOneB.tex", "w")
experimentTwoA = open(f"{directory}/../latex/experimentTwoA.tex", "w")
experimentTwoB = open(f"{directory}/../latex/experimentTwoB.tex", "w")

experimentOneA.write(formatTable(*experimentOneAData, 32))
print("cowabunga")
experimentOneB.write(formatTable(*experimentOneBData, 32))
print("cowabunga")
experimentTwoA.write(formatTable(*experimentTwoAData, 32))
print("cowabunga")
experimentTwoB.write(formatTable(*experimentTwoBData, 32))
print("cowabunga")


experimentOneA.close()
experimentOneB.close()
experimentTwoA.close()
experimentTwoB.close()

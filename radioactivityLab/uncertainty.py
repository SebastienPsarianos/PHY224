import numpy as np
from main import directory, calculate_uncertainty

_, total = np.loadtxt(
    f"{directory}/Barium137_20220929AM_20min20sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)
_, background = np.loadtxt(
    f"{directory}/Background20220929AM-20min20s.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)
mean_background = np.average(background)
sample_count = total - mean_background

_, total_fiesta = np.loadtxt(
    f"{directory}/FiestaPlate(U)-20220929_20min3sec.txt",
    delimiter="\t",
    skiprows=2,
    unpack=True,
)

sample_fiesta = total_fiesta - mean_background


def round_to_one_sig_fig(value):
    num, exp = np.format_float_scientific(value, 0).split(".e")
    return int(num) * 10 ** int(exp)


def formatTable(counts, subtracted, numPerTable, interval):
    countTime = np.arange(interval, 1201, interval)
    i = 0

    if subtracted != []:
        tableStart = "\\begin{tabular}{| p{0.14\\textwidth} | p{0.14\\textwidth} | p{0.14\\textwidth} |}"
        tableStart += f"\\hline\nTime Interval (s) & Total Measured Count & Count without background\\\\\n\\hline\n"

        newString = tableStart

        for j in range(len(counts)):
            if j % numPerTable == 0 and j != 0:
                if i % 2 == 0:
                    newString += "\\hline\n\\end{tabular}\\quad\n"
                else:
                    newString += "\\hline\n\\end{tabular}\\\\\n"

                newString += tableStart
                i += 1

            newString += f"{countTime[j] - interval}-{countTime[j]} & ${int(counts[j])}$ & ${round(subtracted[j])}\\pm {round_to_one_sig_fig(calculate_uncertainty(counts[j], mean_background))}$\\\\\n"

        return newString + "\\hline\n\\end{tabular}"

    tableStart = (
        "\\begin{tabular}{| p{0.14\\textwidth} | p{0.14\\textwidth}|}"
    )
    tableStart += (
        f"\\hline\nTime Interval (s) & Total Measured Count\\\\\n\\hline\n"
    )

    newString = tableStart

    for j in range(len(counts)):
        if j % numPerTable == 0 and j != 0:
            if i % 2 == 0:
                newString += "\\hline\n\\end{tabular}\\quad\n"
            else:
                newString += "\\hline\n\\end{tabular}\\\\\n"

            newString += tableStart
            i += 1

        newString += f"{countTime[j] - interval}-{countTime[j]} & ${int(counts[j])}$\\\\\n"

    return newString + "\\hline\n\\end{tabular}"


fiestaTables = open(f"{directory}/latex/fiestaTables.tex", "w")
bariumTables = open(f"{directory}/latex/bariumTables.tex", "w")
backgroundTables = open(f"{directory}/latex/backgroundTables.tex", "w")

fiestaTables.write(formatTable(total_fiesta, sample_fiesta, 40, 3))
bariumTables.write(formatTable(total, sample_count, 30, 20))
backgroundTables.write(formatTable(background, [], 30, 20))

fiestaTables.close()
bariumTables.close()
backgroundTables.close()

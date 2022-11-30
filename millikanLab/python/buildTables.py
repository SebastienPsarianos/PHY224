from main import *

methodOneTable = open(f"{directory}/../latex/methodOneTable.tex", "w")
methodTwoTable = open(f"{directory}/../latex/methodTwoTable.tex", "w")


def buildTable(charges, chargeUncertainty, method):
    table = "\\begin{tabular}{| p{0.2\\textwidth} | p{0.6\\textwidth} |}\n"
    table += "\\hline\n"
    table += f"Sample & Charge as calculated with {method} (C)\\\\\n"
    table += "\\hline\n"

    for i in range(len(charges)):
        charge, chargeExp = str(charges[i]).split("e")
        uncertainty, uncertaintyExp = str(chargeUncertainty[i]).split("e")
        table += f"${i +  1}$ & ${charge}\\times 10^{{{chargeExp}}} \pm {uncertainty}\\times10^{{{uncertaintyExp}}}$\\\\\n"
    table += "\\hline\n"
    table += "\end{tabular}\n"
    return table


methodOneTable.write(
    buildTable(
        scaledOne / (10 ** scaleFactorOne),
        scaledOneU / (10 ** scaleFactorOneU),
        "method one",
    )
)

methodTwoTable.write(
    buildTable(
        scaledTwo / (10 ** scaleFactorTwo),
        scaledTwoU / (10 ** scaleFactorTwoU),
        "method two",
    )
)


methodOneTable.close()
methodTwoTable.close()

import os

directory = os.path.dirname(os.path.realpath(__file__))

# Replace this with the directory that the files are in
sampleDataFileNames = os.listdir(f"{directory}/rawData/")

samples = []

for fileName in sampleDataFileNames:
    fallingPositions = []
    risingPositions = []
    test = open(f"{directory}/rawData/{fileName}")
    lines = test.readlines()
    for line in lines:
        fallingPosition, risingPosition = line.strip("\n").split(",")
        if fallingPosition != "":
            fallingPositions.append(int(fallingPosition))
        if risingPosition != "":
            risingPositions.append(int(risingPosition))

    test.close()

    samples.append((fallingPositions, risingPositions))
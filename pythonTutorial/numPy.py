import numpy as np

oneToOneHundred = np.arange(0, 100,1)
tenZeros = np.zeros(10, dtype=int)
matrix = np.reshape(oneToOneHundred, (10, 10))

print(oneToOneHundred)
print(tenZeros)
print(matrix * 10)
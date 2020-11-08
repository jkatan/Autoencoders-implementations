import matplotlib
import matplotlib.pyplot as plt
import numpy as np

indexToCharMapper = {
	0: '@',
	1: 'A',
	2: 'B',
	3: 'C',
	4: 'D',
	5: 'E',
	6: 'F',
	7: 'G',
	8: 'H',
	9: 'I'
}

def graph(latentSpaceFilename):
	latentSpaceFilename = open(latentSpaceFilename, "r")
	latentSpaceLines = latentSpaceFilename.readlines()

	latentX = []
	latentY = []

	for line in latentSpaceLines:
		sample = line.split(",")
		latentX.append(float(sample[0]))
		latentY.append(float(sample[1]))

	print("Letter positions in latent space: ")
	for i in range(len(latentX)):
	    x = latentX[i]
	    y = latentY[i]
	    print(indexToCharMapper[i] + ": " + "(" + str(x) + ", " + str(y) + ")")
	    plt.plot(x, y, 'bo')
	    plt.text(x * (1 + 0.05), y * (1 + 0.05) , indexToCharMapper[i], fontsize=10)

	plt.show()

graph("latentSpace.csv")

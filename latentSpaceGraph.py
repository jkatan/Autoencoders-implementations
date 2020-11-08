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

def graph(latentSpaceFilename, closeLettersIndexesFilename, interpolatedPointFilename):
	interpolatedPointFile = open(interpolatedPointFilename, "r")
	interpPointCoordinates = interpolatedPointFile.readlines()[0].split(",")
	interpPointCoordinates[0] = float(interpPointCoordinates[0])
	interpPointCoordinates[1] = float(interpPointCoordinates[1])
	plt.plot(interpPointCoordinates[0], interpPointCoordinates[1], "ro")

	closeLettersIndexesFile = open(closeLettersIndexesFilename, "r")
	closePointsIndexes = closeLettersIndexesFile.readlines()[0].split(",")
	closePointsIndexes[0] = int(closePointsIndexes[0])
	closePointsIndexes[1] = int(closePointsIndexes[1])
	print("Indexes of close points to interpolate: " + str(closePointsIndexes))

	latentSpaceFile = open(latentSpaceFilename, "r")
	latentSpaceLines = latentSpaceFile.readlines()

	latentX = []
	latentY = []

	for line in latentSpaceLines:
		sample = line.split(",")
		latentX.append(float(sample[0]))
		latentY.append(float(sample[1]))

	closePointsX = [latentX[closePointsIndexes[0]], latentX[closePointsIndexes[1]]]
	closePointsY = [latentY[closePointsIndexes[0]], latentY[closePointsIndexes[1]]]
	plt.plot(closePointsX, closePointsY)

	print("Letter positions in latent space: ")
	letterIndex = 0
	for i in range(len(latentX)):
	    x = latentX[i]
	    y = latentY[i]
	    print(indexToCharMapper[i] + ": " + "(" + str(x) + ", " + str(y) + "), index: " + str(letterIndex))
	    plt.plot(x, y, 'bo')
	    plt.text(x * (1 + 0.05), y * (1 + 0.05) , indexToCharMapper[i], fontsize=10)
	    letterIndex += 1

	plt.show()

graph("latentSpace.csv", "closeLettersIndexes.csv", "interpolatedPoint.csv")

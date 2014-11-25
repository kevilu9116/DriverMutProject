"""
This program gathers the following stats for a mutation or expression tumor matrix:
1. Mean and STD of the rowsums of the matrix. This gives us the average number of tumor 
2. Mean and STD of the colsums of the matrix.
"""

import sys
import os
import numpy as np
from NamedMatrix import NamedMatrix

filePath = sys.argv[1]
myFiles = os.listdir(filePath)
for f in myFiles:
	if (".GtM" in f or ".GeM" in f) and ".~lock" not in f:
		myInputMatrix = NamedMatrix(filePath + "/" + f)

		#Get rowsums of tumor, the obtain the mean and standard deviation of the rowsums
		rowSum = myInputMatrix.data.sum(axis=1)
		avgRowSum = sum(rowSum) / len(rowSum)
		stdRowSum = np.std(rowSum)

		#Get colsums of tumor, the obtain the mean and standard deviation of the colsums
		colSum = myInputMatrix.data.sum(axis=0)
		avgColSum = sum(colSum) / len(colSum)
		stdColSum = np.std(colSum)

		#Rank the top 100 colsums
		genes = myInputMatrix.colnames
		tupleList = []
		for i in range(len(colSum)):
			tupleList.append((colSum[i], genes[i]))
		tupleList.sort(reverse=True)
		topTuples = tupleList[0:100]


		if ".GtM" in f:
			outputFile = open(filePath + "/GtMStats.txt", "w")
		elif ".GeM" in f:
			outputFile = open(filePath + "/GeMStats.txt", "w")

		tumorType = f.strip().split(".")[0]
		outputFile.write(tumorType + "," + str(avgRowSum) + "," + str(stdRowSum) + "," + str(len(rowSum)) + "," + str(avgColSum) + "," + str(stdColSum) + "," + str(avgColSum / len(rowSum)) + "\n")
		# outputFile.write("Rowsum Mean: " + str(avgRowSum) + " Rowsum STD: " + str(stdRowSum) + " Length of row: " + str(len(colSum)) + "\n")
		# outputFile.write("Colsum Mean: " + str(avgColSum) + " Colsum STD: " + str(stdColSum) + " Length of columns: " + str(len(rowSum)) + "\n")
		for i in range(len(topTuples)):
			if i == len(topTuples) - 1:
				outputFile.write(topTuples[i][1])
			else:
				outputFile.write(topTuples[i][1] + ",")
		outputFile.close()




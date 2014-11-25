"""
This program makes a frequency matrix depicting the normalized frequencies of each mutation type (1, 2, 3, -2, -3)
for each cancer type and outputs them into a csv file.
"""

import os

gtmDir = "/home/kevin/GroupDropbox/TCI/GtM.withmutinfo"
outputFile = open("/home/kevin/GroupDropbox/TCI/GtM.withmutinfo/BarplotNormalizedMatrix.csv", "w")

outputFile.write("Tumor Type,1,2,3,-2,-3\n")

fileDir = os.listdir(gtmDir)

for f in fileDir:
	tumorType = f.split(".")[0]
	numOnes = 0
	numTwos = 0
	numThrees = 0
	numNegTwos = 0
	numNegThrees = 0

	curMat = open(gtmDir + "/" + f)
	curMat.readline()

	for line in curMat:
		curTumorSample = line.strip().split(",")
		numOnes += sum(1 for i in curTumorSample if i == "1")
		numTwos += sum(1 for i in curTumorSample if i == "2")
		numThrees += sum(1 for i in curTumorSample if i == "3")
		numNegTwos += sum(1 for i in curTumorSample if i == "-2")
		numNegThrees += sum(1 for i in curTumorSample if i == "-3")

	totalNumAlterations = numOnes + numTwos + numThrees + numNegTwos + numNegThrees
	if totalNumAlterations == 0:
		print "No alterations found in " + f
		continue
	percentOnes = float(numOnes) / totalNumAlterations
	percentTwos = float(numTwos) / totalNumAlterations
	percentThrees = float(numThrees) / totalNumAlterations
	percentNegTwos = float(numNegTwos) / totalNumAlterations
	percentNegThrees = float(numNegThrees) / totalNumAlterations
	outputFile.write(tumorType + "," + str(percentOnes) + "," + str(percentTwos) + "," + str(percentThrees) + "," + str(percentNegTwos) + "," + str(percentNegThrees) + "\n")
	curMat.close()

outputFile.close()


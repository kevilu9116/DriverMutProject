from NamedMatrix import *
import os
import sys

# #read list of files of causalscore matrices
filePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data.v2/UCEC/"
fileDir = os.listdir(filePath)

#creaete a dictionary MutGenes
mutGenesDict = {}
gtDriverCount = {}

driverCallThreshold = 3

# #for each file, i.e, each Tumor
for f in fileDir[:10]:

	curMatrix = NamedMatrix(filename = filePath + f)
	gtList = curMatrix.getRownames()
	geList = curMatrix.getColnames()

	#for each Gt, i.e. row
	for i in range(len(gtList)):

		if gtList[i] == "A0":
			continue

		geIndices = np.where(curMatrix.data[i] > .7)[0]

		if len(geIndices) == 0:
			continue

		if len(geIndices) > driverCallThreshold:
			if gtList[i] not in gtDriverCount.keys():
				gtDriverCount[gtList[i]] = 1
			else:
				gtDriverCount[gtList[i]] += 1

		#if the Gt is not in the keys of MutGenes, create entry and assign its value as an empty dictionary
		if gtList[i] not in mutGenesDict.keys():
			mutGenesDict[gtList[i]] = {}
		for index in geIndices:
			geneName = geList[index]
			if geneName not in mutGenesDict[gtList[i]].keys():
				mutGenesDict[gtList[i]][geneName] = 1
			else:
				mutGenesDict[gtList[i]][geneName] += 1

#Sort gtDriverCount by value

sortedGTTuples = sorted(gtDriverCount.items(), key=lambda x:x[1])
sortedGTTuples.reverse()

print "\n"

for gt, counts in sortedGTTuples:
	sortedGEs = sorted(mutGenesDict[gt].items(), key=lambda x:x[1])
	sortedGEs.reverse()
	print gt + " " + str(counts) + " " + str(sortedGEs)


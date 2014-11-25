from NamedMatrix import *
import os
import sys

driverCallThreshold = 5
posteriorThreshold = 0.8
minCoverage = 0.1

# #read list of files of causalscore matrices
tciResultsDir = "/home/projects/TCIResults/Tumor.Type.Data.v5/"
outPutDir = "/home/kevin/projects/TCIResults/DriverFreqAndGeAssociationResults/DriverFreqStatsv5"
tumorTypes = ["PANCAN.with.Sep.TumorType.ID"] #os.listdir(tciResultsDir)
for tumor in tumorTypes:
#	if "~" in tumor or "." in tumor:
#		continue

	tumorFiles = os.listdir(tciResultsDir + tumor)

	#creaete a dictionary MutGenes
	mutGenesDict = {}
	gtDriverCount = {}
	#check if the file has already been written, if it has, skip
	# if os.path.exists("/home/kevin/projects/TCIResults/DriverFreqStatsv5" + "/" + tumor + ".txt"):
	# 	continue

	# #for each file, i.e, each Tumor
	for f in tumorFiles:
		if "~" in f:
			continue

		curMatrix = NamedMatrix(filename = tciResultsDir + tumor + "/" + f)
		if not curMatrix:
			continue
		gtList = curMatrix.getRownames()
		geList = curMatrix.getColnames()

		#for each Gt, i.e. row
		for i in range(len(gtList)):

			if gtList[i] == "A0":
				continue

			geIndices = np.where(curMatrix.data[i] > posteriorThreshold)[0]

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

	#Output the results 
	if not os.path.exists(outPutDir):
		os.mkdir(outPutDir)
	
	# write to file
	outputFile = open(outPutDir + "/" + tumor + ".txt", "w")
	for gt, counts in sortedGTTuples:
		GEs = [x for x in mutGenesDict[gt].items() if (float(x[1])/float(counts)) > minCoverage]
		sortedGEs = sorted(GEs, key=lambda x:x[1])
		sortedGEs.reverse()
		outputFile.write(gt + ";" + str(counts) + ";" + str(sortedGEs) + "\n")
	outputFile.close()



import sys
import os
import numpy as np

# #read list of files of causalscore matrices
filePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data.v2/UCEC/"
fileDir = os.listdir(filePath)

# #creaete a dictionary MutGenes
mutGenesDict = {}

# #for each file, i.e, each Tumor
# gtList = np.genfromtxt(filePath + "TCGA-A5-A1OH.csv", delimiter=',', usecols=0, dtype=str)
# test = np.genfromtxt(filePath + "TCGA-A5-A1OH.csv", delimiter = ",", names = True)[:, 1:]
# #print test[0]
# print test.dtype.names

tumorMatrix = open(filePath + "TCGA-A5-A1OH.csv", "r")
geList = tumorMatrix.readline().strip().split(',')[1:]

for line in tumorMatrix:
	curGT = line.strip().split(',')[0]
	rowData = line.strip().split(',')[1:]
	if curGT not in mutGenesDict.keys():
		mutGenesDict[curGT] = {}
	for i in range(len(rowData)):
		if rowData[i] > .7:
			if geList[i] not in mutGenesDict[curGT].keys():
				mutGenesDict[curGT][geList[i]] = 1
			else:
				mutGenesDict[curGT][geList[i]] += 1



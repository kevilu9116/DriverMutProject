import sys
import os


mutInfoDir = sys.argv[1]
pancanFile = open(sys.argv[2], "r")
outputFile = open(sys.argv[3], "w")

tumorSampleDict = {}
tumorFiles = os.listdir(mutInfoDir)

for t in tumorFiles:
	curFile = open(mutInfoDir + "/" + t, "r")
	genes = curFile.readline().strip().split(",")
	genes.pop(0)
	for line in curFile:
		curLineList = line.strip().split(",")
		curSample = curLineList[0]
		tupleList = []
		for i in range(1, len(curLineList)):
			tupleList.append((genes[i - 1], curLineList[i]))
		tumorSampleDict[curSample] = tupleList
	curFile.close()
print "Done creating dictionary."

pancanMatrix = [[]]
pancanGenes = pancanFile.readline().strip().split(",")
pancanMatrix.append(pancanGenes)
pancanGenes.pop(0)

for line in pancanFile:
	curTumorID = line.strip().split(",")[0]
	curTumorIDDataList = [curTumorID]
	curMutData = tumorSampleDict[curTumorID]
	for gene in pancanGenes:
		flag = 0
		for item in curMutData:
			if gene == item[0]:
				curTumorIDDataList.append(item[1])
				flag = 1
				break
		if flag == 0:
			curTumorIDDataList.append("0")
	pancanMatrix.append(curTumorIDDataList)

print "Done making PANCAN matrix."
pancanFile.close()

for line in pancanMatrix:
	for i in range(len(line)):
		if i == len(line) - 1:
			outputFile.write(str(line[i]) + "\n")
		else:
			outputFile.write(str(line[i]) + ",")
outputFile.close()



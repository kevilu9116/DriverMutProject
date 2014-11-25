import sys
import os

inputDir = sys.argv[1]
dataType = sys.argv[2] #either "GtM" or "GeM"
outputFile = open(sys.argv[3], "w")

outputFile.write("Tumor Type,Mean SGA/Tumor,SD of SGA/Tumor,Total Tumors,Avg Tumors/SGA,SD Tumors/SGA,Avg Percent Tumors/SGA\n")

tumorTypes = os.listdir(inputDir)
for tumor in tumorTypes:
	if "." in tumor:
		continue
	tumorFiles = os.listdir(inputDir + "/" + tumor)
	dataFile = "null"

	for f in tumorFiles:
		if dataType + "Stats" in f:
			dataFile = open(inputDir + "/" + tumor + "/" + f, "r")
			break
	if dataFile == "null":
		print "Error. Stats file not found."
		continue
	outputFile.write(dataFile.readline())
	dataFile.close()
outputFile.close()



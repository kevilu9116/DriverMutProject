import os

folderList = os.listdir("/home/kevin/GroupDropbox/TCI/Tumor.Type.Data")
for cancer in folderList:
	if not "." in cancer:
		cancerFiles = os.listdir("/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer)
		mutMatrixFilePath = "null"
		degMatrixFilePath = "null"
		for i in cancerFiles:
			if "GtM.csv" in i:
				mutMatrixFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer + "/" + i
			if "GeM.csv" in i:
				degMatrixFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer + "/" + i
		outputFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer + "/CombLogicAND.Results"
		print mutMatrixFilePath + "\t" + degMatrixFilePath + "\t" + outputFilePath


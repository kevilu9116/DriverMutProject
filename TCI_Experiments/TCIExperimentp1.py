from SparseMatrixTCI import *

geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

#mutMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GtM.csv"
#degMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GeM.csv"


fileDir = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5"
tumorFiles = ['PANCAN']
#tumorFiles = ['BRCA', 'KICH', 'COAD', 'GBM']
#tumorFiles = ['HNSC', 'KIRC', 'KIRP', 'LUAD', 'UCEC']
#tumorFiles = ['LUSC', 'OV', 'PAAD', 'PANCAN']
#tumorFiles = ['PRAD', 'READ', 'SKCM', 'STAD', 'THCA']

for f in tumorFiles:
	mutMatrixFilePath = fileDir + "/" + f + "/" + f + ".GtM.csv"
	degMatrixFilePath = fileDir + "/" + f + "/" + f + ".GeM.csv"
	outputPath = "/home/kevin/projects/TCIResults/Tumor.Type.Data.v5/" + f
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, outputPath = outputPath,  dictGeneLength = geneLengthDict)

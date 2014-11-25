from SparseMatrixTCI import *
from NamedMatrix import NamedMatrix
from utilTCI import parseGeneLengthDict 
import os

geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

mutMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN/PANCAN.GtM.csv"
degMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN/PANCAN.GeM.csv"

fileDir = "/home/kevin/projects/TCIResults/Tumor.Type.Data.v5/PANCAN/"
if not os.path.exists(fileDir):
	os.path.mkdir(fileDir)

calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, outputPath = fileDir, v0=0.4, dictGeneLength = geneLengthDict, rowBegin = 4001, rowEnd = None)
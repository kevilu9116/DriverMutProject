from PanCanSparseMatrixTCI import *
from NamedMatrix import NamedMatrix
from utilTCI import parseGeneLengthDict 
import os

geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

mutMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/PANCAN.GtOnlyM.csv"
degMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/PANCAN.GeM.csv"
tumorTypePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/PANCAN.tumor.type.indicator.matrix.csv"
    
outputFilePath = "/home/projects/TCIResults/Tumor.Type.Data.v5/PANCAN.with.Sep.TumorType.ID"

calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, tumorTypeFN = tumorTypePath, PANCANFlag = 1, outputPath = outputFilePath,  dictGeneLength = geneLengthDict, v0 = 0.3, rowBegin = 4001, rowEnd = 5195)
    


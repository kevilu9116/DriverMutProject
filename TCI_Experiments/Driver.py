from SparseMatrixTCI import calcTCI
import theano.tensor as T
from theano import function, shared,  config
import numpy as np
import sys
import math
import os
from NamedMatrix import NamedMatrix
import scipy as s
import scipy.sparse as sp
from util import * 

geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

mutMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GtM.csv"
degMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GeM.csv"


#    mutMatrixFilePath = "/home/kevin/GroupDropbox/TCI/chunhui.testmatrices/GtM.testset.csv"
#    degMatrixFilePath = "/home/kevin/GroupDropbox/TCI/chunhui.testmatrices/GeM.testset.csv"
#    outputFilePath = "/home/kevin/GroupDropbox/TCI/chunhui.testmatrices/OctTestResults"

#    mutMatrixFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/SKCM/SKCM.GtM.csv"
#    degMatrixFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/SKCM/SKCM.GeM.csv"
#    outputFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/SKCM/SKCMSparseGPUTest"

# folderList = os.listdir("/home/kevin/projects/TCIResults/Tumor.Type.Data")
# for cancer in folderList:
#     if not "." in cancer:
#         cancerFiles = os.listdir("/home/kevin/projects/TCIResults/Tumor.Type.Data/" + cancer)
#         mutMatrixFilePath = "null"
#         degMatrixFilePath = "null"
#         outputFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/" + cancer + "/CombLogicAND.Results"
#         if os.listdir(outputFilePath) != []:
#             continue
#         for i in cancerFiles:
#             if "GtM.csv" in i:
#                 mutMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/" + cancer + "/" + i
#             if "GeM.csv" in i:
#                 degMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/" + cancer + "/" + i

        #Calculate TCI Score by calling calcTCI with the following arguments:
        #mutation matrix, DEG matrix, output filepath, gene length dictionary, and an optional operation flag

priors = [1.1754943508222875e-38, 0.1, 0.3, 0.4, 0.5]
fileDir = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/SingleGtM-Prior"
#outputFilePaths = [fileDir + str(x) for x in priors]  #another way to 
outputFilePaths = [fileDir + "SingleGtM0.0", fileDir + "SingleGtM0.1", fileDir + "SingleGtM0.3", fileDir + "SingleGtM0.4", fileDir + "SingleGtM0.5"]
for i in range(len(priors)):
    # create the directory if necessary
    if not os.path.exists(outputFilePaths[i]):
        os.makedirs(outputFilePaths[i])

    calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, outputPath = outputFilePaths[i],  dictGeneLength = geneLengthDict, v0 = priors[i])

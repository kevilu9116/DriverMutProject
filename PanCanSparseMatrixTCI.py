# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 22:03:45 2014

@author: Kevin Lu

This module contains functions required for performing the tumor-specific 
causal inference (TCI).  See Technical report "A method for identify driver
genomic alterations at the individual tumor level" by Cooper, Lu, Cai and Lu.

This is a version that specifically designed to model Pan Cancer TCI, which 
include a tissue type indicator as an candidate to explain certain tissue-specific 
confounding factors

 
"""
from utilTCI import *
import math
import os
from NamedMatrix import NamedMatrix
import scipy as s
import scipy.sparse as sp

###############################################################################################
###### module-wide constances  ######
minNumSGA = 5

# Theano matrix addition
# sum matrix ops

m1 = T.fmatrix()
m2 = T.fmatrix()
add = function([m1, m2], m1 + m2, allow_input_downcast=True)

def calcTCI (mutcnaMatrixFN, degMatrixFN, tumorTypeFN = None, alphaNull = [1, 1], alphaIJKList = [2, 1, 1, 2], v0=0.2, 
             dictGeneLength = None, outputPath = ".", opFlag = None, PANCANFlag = None, rowBegin=0, rowEnd = None):
    """ 
    calcTCI (mutcnaMatrix, degMatrix, alphaIJList, alphaIJKList, dictGeneLength)
    
    Calculate the causal scores between each pair of SGA and DEG observed in each tumor
    
    Inputs:
        mutcnaMatrixFN      A file containing a N x G binary matrix containing the mutation and CNA 
                            data of all tumors.  N is the number of tumors and 
                            G is number of total number of unique genes.  For a
                            tumor, genes that have SGAs are indicated by "1"s and "0" 
                            otherwise. 
                            Note the last 19 columns are indicators of the tumor 
        degMatrixFN         A file contains a N x G' binary matrix representing DEG
                            status.  A "1" indicate a gene is differentially expressed
                            in a tumor.
                            
        tumorTypeFN         A string of filename.  The file contains N x T matrix, in which
                            each row only has one element set to 1, rest to zero, as an indicator
                            which type of cancer each tumor belongs to 
        
        alphaIJList         A list of Dirichlet hyperparameters defining the prior
                            that a mutation event occurs
                            
        alphaIJKList        A list of Dirichlet hyperparameters for caulate the prior
                            of condition probability parameters. alphaIJK[0]: mut == 0 && deg == 0;
                            alphaIJK[1]: mut == 0 && deg == 1; alphaIJK[2]: mut == 1 && deg == 0;
                            alphaIJK[3]: mut == 1 && deg == 1
                            
        v0                  A float scalar indicate the prior probability that a DEG
                            is caused by a non-SGA factor 
                            
        PANCANFlag          A boolean flag to indicate if we are doing PANCAN
        dictGeneLength      A dictionary keeps the length of each of G genes in the 
                            mutcnaMatrix
	
	rowBegin, rowEnd        These two arguments control allow user to choose which block out of all tumors (defined by the two 
			    row numbers) will be processes in by this function.  This can be used to process
			    mulitple block in a parallel fashion.
    """
    # check if gene length dictionary is set
    if not dictGeneLength :
        print "Gene length dictionary not provided, quit\n"
        sys.exit()
    
    # read in data in the form of NamedMatrix 
    try:
        mutcnaMatrix  = NamedMatrix(mutcnaMatrixFN)
    except:
        print "Failed to import data matrix %s\n" % mutcnaMatrixFN
        sys.exit() 
        
    try:
        degMatrix = NamedMatrix(degMatrixFN)
    except:
        print "Failed to import data matrix %s\n" % degMatrixFN
        sys.exit()
    
    mutGeneNames = mutcnaMatrix.getColnames()
    mutTumorNames = mutcnaMatrix.getRownames()
    degGeneNames = degMatrix.getColnames()
    exprsTumorNames = degMatrix.getRownames()
    
    #check if same tumor names from two matrices above agree
    if exprsTumorNames != mutTumorNames:
        print "The tumors for mutcnaMatrix and degMatrix do not fully overlap!"
        print degMatrix.getRownames()
        print mutcnaMatrix.getRownames()
        sys.exit()

    tumorNames = exprsTumorNames
    nTumors, nMutGenes = mutcnaMatrix.shape()
    
    # now perform PANCAN analysis related tasks
    if PANCANFlag: 
        if not tumorTypeFN:
            print "Cannot perform PANCAN analysis without tumor-type-indicator matrix"
            sys.exit()
        try: 
            tumorTypeMatrix = NamedMatrix(tumorTypeFN)
        except:
            print "Failed to import tumor type file %s" % tumorTypeFN
            sys.exit()            
        tumorTypeTumorNames = [x.replace("\"", "") for x in tumorTypeMatrix.getRownames()]
        if exprsTumorNames != tumorTypeTumorNames:
            print "The tumors for tumorTypeMatrix and degMatrix do not fully overlap!"
            sys.exit()

        tumorTypes = tumorTypeMatrix.getColnames()    
        # Calculate the prior probability that a tumor-type variable may influence a DEG
        # to be proportional to the number of tumors from a given type
        vt = np.sum(tumorTypeMatrix.data, 0)  # perform a rowsum to count  each type tumor
        vtprior = np.divide(vt, float(nTumors)) 
        
    # Now start looping through a chunk of individual tumors and calculate the causal scores between each pair of SGA and DEG    
    print "Done with loading data, start processing tumor " + str(rowBegin)
    if not rowEnd:
        rowEnd = nTumors - 1
    else:
        if rowEnd >= nTumors:
		rowEnd = nTumors - 1
	elif rowEnd < rowBegin:
            print "Invalid rowEnd < rowBegin arguments given."
            sys.exit()

    if rowBegin > rowEnd:
        print "Invlid rowBegin > rowEnd argument given."
        sys.exit()

    for t in range(rowBegin, rowEnd):
        print "processign tumor  " + tumorNames[t]
        #print pacifier
        if t % 50 == 0:
            print "Processed %s tumors" % str(t)
        
        # collect data related to DEGs to construct a submatrix containing only DEG of the tumor
        degGeneIndx = [i for i, j in enumerate(degMatrix.data[t,:]) if j == 1]
        tumorDEGGenes = [degGeneNames[i] for i in degGeneIndx] 
        tumorDEGMatrix = degMatrix.data[:,degGeneIndx]
 
        # extract the sub-matrix of mutcnaMatrix that only contain the genes that are mutated in a given tumor t
        tumormutGeneIndx = [i for i, j in enumerate(mutcnaMatrix.data[t,:]) if j == 1]        
        tumorMutMatrix = mutcnaMatrix.data[:,  tumormutGeneIndx]
        tumorMutGenes=  [mutGeneNames[i] for i in tumormutGeneIndx] 
        nTumorMutGenes = len(tumorMutGenes)
        
        # Include the tumor-type label into the tumorMutMatrix as a tissue-specific 
        # fake Gt to capture the DEGs that has tissue-specific characterisitics 
        if PANCANFlag:
            tumorTypeLabelIndx = np.where(tumorTypeMatrix.data[t,:] == 1)[0]
            if len(tumorTypeLabelIndx) != 1:
                raise Exception("Fail to extract tumor type")  
            # add the label to the tumorMutGenes
            tumorMutMatrix = np.hstack((tumorMutMatrix, tumorTypeMatrix.data[:,tumorTypeLabelIndx]))                  
            tumorTypeName = tumorTypes[tumorTypeLabelIndx]        
            tumorMutGenes.append(tumorTypeName) 
            nTumorMutGenes = len(tumorMutGenes)
            
        # calculate single pairwise likelihood that an SGA causes a DEG.  Return a matrix where rows are mutGenes, 
        # columns are DEGs, currently without the joint impact
        tumorLnFScore = calcF(tumorMutMatrix, tumorDEGMatrix,  alphaIJKList)

        # If PANCAN analysis, construct combinations of tumor-type label with different GTs to determine the 
        # likelihood of DEG jointly conditioning on GT and tumor-type label.  This enables us to capture
        # the fact that a GT regulate a GE but they also have a high tendency in co-occurring in a specific tumor type
        if PANCANFlag:              
            # Now, calcuate the log likelihood of joint impact of tumor label with individual GTs on each GE
            jointGTandTumorLableFScore = np.zeros((tumorMutMatrix.shape[1], tumorDEGMatrix.shape[1])) 
                
            # GT == 1 && Label == 1.  Use mulitplication as AND operation
            tmpMutMatrix = np.multiply(tumorMutMatrix, tumorTypeMatrix.data[:, tumorTypeLabelIndx])  
            tumorLnFScore = calcF(tmpMutMatrix, tumorDEGMatrix,  alphaIJKList)
            jointGTandTumorLableFScore = add(jointGTandTumorLableFScore, tumorLnFScore)
            
            # GT == 1 && label == 0
            tmpMutMatrix = np.multiply(tumorMutMatrix, tumorTypeMatrix.data[:, tumorTypeLabelIndx]==0) 
            tumorLnFScore = calcF(tmpMutMatrix, tumorDEGMatrix,  alphaIJKList)
            jointGTandTumorLableFScore = add(jointGTandTumorLableFScore, tumorLnFScore)

            # GT == 0 && label == 1
            tmpMutMatrix = np.multiply(tumorMutMatrix == 0, tumorTypeMatrix.data[:, tumorTypeLabelIndx])
            tumorLnFScore = calcF(tmpMutMatrix, tumorDEGMatrix,  alphaIJKList)  
            jointGTandTumorLableFScore = add(jointGTandTumorLableFScore, tumorLnFScore)
            
            # GT == 0 && label == 0
            tmpMutMatrix = np.multiply(tumorMutMatrix == 0, tumorTypeMatrix.data[:, tumorTypeLabelIndx] == 0) 
            tumorLnFScore = calcF(tmpMutMatrix, tumorDEGMatrix,  alphaIJKList)  
            jointGTandTumorLableFScore = add(jointGTandTumorLableFScore, tumorLnFScore)

            # stack the the joint loglikelihood matrix on top to the tumorLnFScore.  
            #Remove the tumor-type label variable from the matrix derived from tumorMutMatrix
            tumorLnFScore = np.vstack((jointGTandTumorLableFScore[:-1,:] , tumorLnFScore))             

        # Calculate the likelihood that A0, which is 1 for all tumors, as a cause for DEGs.  
        # Then, stack to the LnFScore, equivalent to adding a column of '1' to 
        # represent the A0 in tumorMutMatrix
        nullFscore = calcNullF(tumorDEGMatrix, alphaNull)
        tumorLnFScore = np.vstack((tumorLnFScore, nullFscore)) 

        # calcualte the prior probability that any of mutated genes plus A0 can be a cause for a DEG.
        if PANCANFlag:
            lntumorMutPriors = calcPanCanLnPrior(tumorMutGenes, dictGeneLength, vtprior[tumorTypeLabelIndx], v0)
        else:
            lntumorMutPriors = calcLnPrior(tumorMutGenes, dictGeneLength, v0)
        tumorLnFScore = np.add(tumorLnFScore.T, lntumorMutPriors).T  # add to each column, note double transposes because  numpy broadcasts by row
               
        # calculate the normalizer for each column (GE).  
        colLogSum = calcColNormalizer(tumorLnFScore)       
        normalizer = np.tile(colLogSum, (tumorLnFScore.shape[0], 1))    
        posteriorAll = np.exp(add(tumorLnFScore, - normalizer))
        
        # now sum the posterior of each single GT with the posteriors of joint GT-Tumor-Type  
        posterior = np.add(posteriorAll[0:nTumorMutGenes-1, :], posteriorAll[nTumorMutGenes - 1:-2, :])
        posterior = np.vstack((posterior, posteriorAll[-2:, :]))        
        
        #write out the results 
        tumorMutGenes.append('A0')
        tumorPosterior = NamedMatrix(npMatrix = posterior, rownames = tumorMutGenes, colnames = tumorDEGGenes)
        tumorPosterior.writeToText(filePath = outputPath, filename = tumorNames[t] + ".csv")

        
def calcNullF(degMatrix, alphaNull):
    """
    This funciton calculate the terms in equation #7 of white paper for 
    the leak cause node A0 (A-null), which only require 3 terms because the cause exists 
    for every tumor.  A special prior is a special set of hyperparameter  
    
    """
    N = degMatrix.shape[0]
    # because all tumors have A0 set to 1
    term1 = s.special.gammaln (sum(alphaNull)) - s.special.gammaln(sum(alphaNull) + N) 
    term2 = map(s.special.gammaln, degMatrix.sum(axis = 0) + alphaNull[1]) - s.special.gammaln(alphaNull[1])
    term3 = map(s.special.gammaln, (degMatrix==0).sum(axis= 0) + alphaNull[0]) - s.special.gammaln(alphaNull[0])
    return  np.array(term1 + term2 + term3)
        
        
###################################################################

###  The following are Theano represeantion of certain functions

# declare a function that calcualte gammaln on a shared variable on GPU
aMatrix = shared(np.zeros((65536, 8192)), config.floatX, borrow=True)
gamma_ln = function([ ], T.gammaln(aMatrix))
theanoExp = function([ ], T.exp(aMatrix))
    
alpha = T.fscalar()
gamma_ln_scalar = function([alpha], T.gammaln(alpha), allow_input_downcast=True)

# now compute the second part of the F-score, which is the covariance of mut and deg
mutMatrix = shared(np.ones((32768, 4096)), config.floatX, borrow=True )  
expMatrix = shared(np.ones((8192, 4096)), config.floatX, borrow=True)
mDotE = function([], T.dot(mutMatrix, expMatrix))

nijk_11 = shared(np.zeros((32768, 4096)), config.floatX)
nijk_01 = shared(np.zeros((32768, 4096)), config.floatX)

fscore = shared(np.zeros((32768, 4096)), config.floatX)
tmpLnMatrix = shared(np.zeros((32768, 4096)), config.floatX, borrow=True)
accumAddFScore = function([], fscore + tmpLnMatrix)

## create 32bit theano copies of mutcan and DEG matrice, make them accessable to GPU
#mutcnaMatrix = shared(np.zeros((8192, 8192 )), config.floatX)
#degMatrix =  shared(np.zeros((8192, 8192 )), config.floatX)

############################################################################################################

def calcF(mutcnaInputMatrix, degInputMatrix, alphaIJKList):

    """
    This function calculate log funciton of the Eq 7 of TCI white paper 

    Input:  mutcnaInputMatrix      A N x m numpy matrix containing mutaiton and CNA data of N tumors and m genes
            degInputMatrix         A N x n numpy matrix containing DEGs from N tumors and d genes
               
            alphaIJList     A list of two elements containing the hyperparameter define the prior distribution for mutation events
            alphaIJKList    A list of four elements containing the hyperparameters defining the prior distribution of condition prability

    Output: A m x d matrix, in which each element contains the F-score of a pair of mutation and DEGs

    F-Score is calcaulated using the following equation.  \frac {\Gamma(\alpha_{ij})} {\Gamma(\alpha_{ij}+ N_{ij})}

    """
    
    #Initialize fscore matrix to zero
    fscore.set_value(np.zeros((mutcnaInputMatrix.shape[1], degInputMatrix.shape[1])), config.floatX)
    # add check if mutcnaMatrix degMatrix is an instance of numpy float matrix of 32 bit

    # calculate the first part of the F-scores, which collect total counts of Gt across tumors
    ni0_vec = np.sum(mutcnaInputMatrix== 0, axis = 0) + alphaIJKList[0] + alphaIJKList[1] # a vector of length m contains total number of cases in which m-th element are ZERO
    ni1_vec = np.sum (mutcnaInputMatrix, axis = 0 ) + alphaIJKList[2] + alphaIJKList[3]  # a vector of length m contains total number cases in which m-th element are ONE
   
    # make a m x n matrix where a m-dimension vectior is copied n times
    aMatrix.set_value(np.tile(ni1_vec, (degInputMatrix.shape[1], 1)).T , config.floatX)
    tmpLnMatrix.set_value(gamma_ln_scalar(alphaIJKList[2] + alphaIJKList[3]) -  gamma_ln(), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)   
   
    aMatrix.set_value(np.tile(ni0_vec, (degInputMatrix.shape[1], 1)).T, config.floatX)    
    tmpLnMatrix.set_value(gamma_ln_scalar(alphaIJKList[0] + alphaIJKList[1]) - gamma_ln(), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)
 
    # calcuate the second term of the eq 7 which has 4 combinations of Gt-vs-GE
    # calc count of mut == 1 && deg == 1 
    # use sparse matrix to save computation
    mutcnaMatrix = sp.csr_matrix(mutcnaInputMatrix.T, dtype = np.float32)
    degMatrix = sp.csc_matrix (degInputMatrix, dtype = np.float32)
    mutDotDeg = mutcnaMatrix.dot(degMatrix).todense()

    aMatrix.set_value(mutDotDeg + alphaIJKList[3], config.floatX)
    nijk_11.set_value(aMatrix.get_value(), config.floatX) 
    tmpLnMatrix.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[3]), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)

    # calc mut == 1 && deg == 0, the latter is not sparse
    mutDotDeg = mutcnaMatrix.dot(degInputMatrix==0)

    aMatrix.set_value(mutDotDeg + alphaIJKList[2], config.floatX)
    tmpLnMatrix.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[2]), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)
    #nijk_10 = shared(mDotE() + alphaIJKList[2], config.floatX)

    # calc mut == 0 && deg == 0, two dense matrices, use Theano and GPU to calculate dot product
    mutMatrix.set_value(mutcnaInputMatrix.T == 0, config.floatX)
    expMatrix.set_value(degInputMatrix==0,config.floatX )

    aMatrix.set_value(mDotE() + alphaIJKList[0], config.floatX)
    tmpLnMatrix.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[0]), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)

    # calc mut == 0 && deg == 1, the deg is a sparse matrix
    degMatrix = sp.csc_matrix (degInputMatrix.T, dtype = np.float32)
    mutDotDeg = degMatrix.dot(mutcnaInputMatrix==0)
    
    aMatrix.set_value(mutDotDeg.T + alphaIJKList[1], config.floatX)
    nijk_01.set_value(aMatrix.get_value(), config.floatX)
    tmpLnMatrix.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[1]), config.floatX)
    fscore.set_value(accumAddFScore(), config.floatX)

    # now caluc the theano final
    fvalues = fscore.get_value()

    # check if the probability that mut == 1 && deg == 1 is bigger than mut == 0 && deg == 1, 
    # if not, set the likelihood that mutated gene is a cause to zero
    condMutDEG_11 = nijk_11.get_value().T / ni1_vec   
    condMutDEG_01 = nijk_01.get_value().T / ni0_vec  
    elementsToSetZero = np.where(condMutDEG_11.T <= condMutDEG_01.T)
    fvalues[elementsToSetZero] = np.NINF  # equivalent to set the element to 0 when exp or logSum 
     
    return fvalues
 

def calcLnPrior(geneNames, dictGeneLength,  v0 = 0.2):
    """
    calLnPrior(geneNames, dictGeneLength, v0)
    
    Input: 
        geneNames       A list of SAG-affected genes that are altered in a give tumor
        dictGeneLength  A dictionary contain the length of all genes
        v0              A weight of a "leak node" besides SGA-affected genes 
                        that may contribute to the differential expression of a gene

    Output:
        lnprior         A list of prior probability values (natural logged) for each given gene
                        
    """
    if v0 >= 1.0 :
        raise Exception ("Exception from calLnPrior:  v0 >= 1.0")
        
    #extract gene lengths for all the genes in 'geneNames'
    listGeneLength = [dictGeneLength[g]  for g in geneNames]
    #Calculate the prior probability by taking each ###########FINISH THIS COMMENT
    inverseLength = [1 / float(x) for x in listGeneLength]
    sumInverseLength = sum(inverseLength)
    prior =  [(1 - v0) * x / sumInverseLength for x in inverseLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior 


def calcPanCanLnPrior (geneNames, dictGeneLength,  vtprior, v0 = 0.2):
    """
    calPanCanLnPrior(geneNames, dictGeneLength, v0)
    
    Input: 
        geneNames       A list of SAG-affected genes that are altered in a give tumor
        dictGeneLength  A dictionary contain the length of all genes
        vt              A weight for tumor type label as potential factor influencign gene expression
        v0              A weight of a "leak node" besides SGA-affected genes 
                        that may contribute to the differential expression of a gene

    Output:
        lnprior         A list of prior probability values (natural logged) for each given gene
                        
    """
    if v0 >= 1.0 :
        raise Exception ("Exception from calLnPrior:  v0 > 1.0")
    elif  vtprior > 1.0:
        raise Exception (vtprior > 1.0)
        
    if v0 < 0 or vtprior < 0:
        raise Exception ("vt or v0 < 0")
        
    #extract gene lengths for all the genes in 'geneNames'
    listGeneLength = [dictGeneLength[g]  for g in geneNames]
    inverseGeneLength = [1 / float(x) for x in listGeneLength]
    inverseGeneLength = [float(x) * vtprior for x in inverseGeneLength[:-1]] + inverseGeneLength 
    sumInverseLength = sum(inverseGeneLength)
    prior =  [(1 - v0) * x / sumInverseLength for x in inverseGeneLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior 
    
    
def calcLnCombPrior(combGeneNames, geneLengthDict, v0):
    """
    calLnCombPrior(geneNames, dictGeneLength, v0)
    
    Input: 
        combGeneNames   A list of combined SAG-affected genes that are altered in a give tumor
        dictGeneLength  A dictionary contain the length of all genes
        v0              A weight of a "leak node" besides SGA-affected genes 
                        that may contribute to the differential expression of a gene

    Output:
        lnprior         A list of prior probability values (natural logged) for each given gene combination

    """

    listGeneLength = []

    #extract gene lengths for each gene combination. Gene lengths for combined genes are simply the sum of
    #the two individual gene lengths.     
    for name in combGeneNames:
        #print combGeneNames
        gene1, gene2 = name.split("/")
        totalLength = float(geneLengthDict[gene1]) + float(geneLengthDict[gene2])
        listGeneLength.append(totalLength)
    
    #Calculate the prior probability by taking each ###########FINISH THIS COMMENT    
    inverseLength = [1 / float(x) for x in listGeneLength] 
    sumInverseLength = sum(inverseLength)
    prior =  [(1-v0)* x / sumInverseLength for x in inverseLength[:-1]] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior 
    
    
def main():
    
    geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

    mutMatrixFilePath = "/home/kevin/Dropbox (XinghuaLu)/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/GtM.test.csv"
    degMatrixFilePath = "/home/kevin/Dropbox (XinghuaLu)/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/GeM.test.csv"
    tumorTypePath = "/home/kevin/Dropbox (XinghuaLu)/Chunhui Collab/Tumor.Type.Data/Modelinputdata.v5/PANCAN.with.Sep.Tumor.ID.Matrix/tumorType.test.csv"
    
    outputFilePath = "/home/projects/TCIResults/Tumor.Type.Data.v5/PANCAN.with.Sep.TumorType.ID"

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
    
    #         #Calculate TCI Score by calling calcTCI with the following arguments:
    #         #mutation matrix, DEG matrix, output filepath, gene length dictionary, and an optional operation flag
    calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, tumorTypeFN = tumorTypePath, PANCANFlag = 1, 
            outputPath = outputFilePath,  dictGeneLength = geneLengthDict, rowBegin = 1, rowEnd = 10, v0 = 0.2)#, opFlag = AND)
    


if __name__ == "__main__":
    main()       


   
    
  
   
   
 

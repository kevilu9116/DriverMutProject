# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 22:03:45 2014

@author: Kevin Lu

This module contains functions required for performing the tumor-specific 
causal inference (TCI).  See Technical report "A method for identify driver
genomic alterations at the individual tumor level" by Cooper, Lu, Cai and Lu.

 
"""

import theano.tensor as T
from theano import function, shared,  config
from PPI_neighbor_dictionary import *
import numpy as np
import sys
import math
import os
from NamedMatrix import NamedMatrix
import scipy as s
import scipy.sparse as sp
from utilTCI import *

###############################################################################################
"""
 The following block define the Theano functions that going to be used in the regular python funcitons
 """
"""
1.  Calculate ln(X + Y) based on ln(X) and ln(Y) using theano library

"""
########### Theano function for calculating logSum
maxExp = -4950.0 
x, y = T.fscalars(2)

yMinusx = y - x  ## this part is for the condition which x > y
xMinusy = x - y  # if x < y
bigger = T.switch(T.gt(x, y), x, y)
YSubtractX = T.switch(T.gt(x,y), yMinusx, xMinusy)       
x_prime =  T.log(1 + T.exp(YSubtractX)) + bigger
calcSum = T.switch(T.lt(YSubtractX, maxExp), bigger, x_prime)
logSum = function([x, y], calcSum, allow_input_downcast=True)


####### end of logSum  ###############

def calcTCI (mutcnaMatrixFN, degMatrixFN, alphaNull = [1, 1], alphaIJKList = [2, 1, 1, 2], v0=0.2, dictGeneLength = None, outputPath = ".", opFlag = None, rowBegin=0, rowEnd = None):
    """ 
    calcTCI (mutcnaMatrix, degMatrix, alphaIJList, alphaIJKList, dictGeneLength)
    
    Calculate the causal scores between each pair of SGA and DEG observed in each tumor
    
    Inputs:
        mutcnaMatrixFN      A file containing a N x G binary matrix containing the mutation and CNA 
                            data of all tumors.  N is the number of tumors and 
                            G is number of total number of unique genes.  For a
                            tumor, genes that have SGAs are indicated by "1"s and "0" 
                            otherwise. 
        degMatrixFN         A file contains a N x G' binary matrix representing DEG
                            status.  A "1" indicate a gene is differentially expressed
                            in a tumor.
        
        alphaIJList         A list of Dirichlet hyperparameters defining the prior
                            that a mutation event occurs
                            
        alphaIJKList        A list of Dirichlet hyperparameters for caulate the prior
                            of condition probability parameters. alphaIJK[0]: mut == 0 && deg == 0;
                            alphaIJK[1]: mut == 0 && deg == 1; alphaIJK[2]: mut == 1 && deg == 0;
                            alphaIJK[3]: mut == 1 && deg == 1
                            
        v0                  A float scalar indicate the prior probability that a DEG
                            is caused by a non-SGA factor 
        
        dictGeneLength      A dictionary keeps the length of each of G genes in the 
                            mutcnaMatrix
	
	rowBegin, rowEnd        These two arguments control allow user to choose which block out of all tumors (defined by the two 
			    row numbers) will be processes in by this function.  This can be used to process
			    mulitple block in a parallel fashion.
    """
    
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
        
    exprsTumorNames = [x.replace("\"", "") for x in degMatrix.getRownames()]
    mutTumorNames = [x.replace("\"", "") for x in mutcnaMatrix.getRownames()]
    if exprsTumorNames != mutTumorNames:
        print "The tumors for mutcnaMatrix and degMatrix do not fully overlap!"
        print degMatrix.getRownames()
        print mutcnaMatrix.getRownames()
        sys.exit()
    
    if  not dictGeneLength :
        print "Gene length dictionary not provided, quit\n"
        sys.exit()
        
    # now we iterate through each tumor to infer the causal relationship between each 
    # pair of mut - deg
    tumorNames = degMatrix.getRownames()
    nTumors, nMutGenes = mutcnaMatrix.shape()
    
    mutGeneNames = mutcnaMatrix.getColnames()
    degGeneNames = degMatrix.getColnames()
    
    # loop through individual tumors and calculate the causal scores between each pair of SGA and DEG    
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

    print "Done with loading data, start processing tumor " + str(rowBegin)
    for t in range(rowBegin, rowEnd):
        #print pacifier
        if t % 50 == 0:
            print "Processed %s tumors" % str(t)
        
        # collect data related to mutations
        tumormutGeneIndx = [i for i, j in enumerate(mutcnaMatrix.data[t,:]) if j == 1]
        if len(tumormutGeneIndx) < 2:
            print tumorNames[t] + " has less than 2 mutations, skip."
            continue
        tumorMutGenes = [mutGeneNames[i] for i in tumormutGeneIndx]        
      
        #now extract the sub-matrix of mutcnaMatrix that only contain the genes that are mutated in a given tumor t
        # stack a column of '1' to represent the A0.          
        tumorMutMatrix = mutcnaMatrix.data[:,  tumormutGeneIndx]
        
        # check if special operations to create combinations of SGA events are needed.  If combination operation is needed, new combined muation matrix 
        # will be created         
        if opFlag == AND:
            tmpNamedMat = NamedMatrix(npMatrix = tumorMutMatrix, colnames = tumorMutGenes, rownames = tumorNames)
            tumorNamedMatrix = createANDComb(tmpNamedMat, opFlag)
            if not tumorNamedMatrix:  # this tumor do not have any joint mutations that is oberved in 2% of all tumors
                continue
            tumorMutMatrix = tumorNamedMatrix.data
            tumorMutGenes = tumorNamedMatrix.colnames
           
        ## check operation options:  1) orginal, do nothing and contiue
        # otherwise creat combinary matrix using the tumorMutMatrix 
        # createANDCombMatrix(tumorMutMatrix, operationFlag)
        if not opFlag:
            lntumorMutPriors = calcLnPrior(tumorMutGenes, dictGeneLength, v0)  # a m-dimension vector with m being number of mutations
        else:
            #print tumorMutGenes[:10]
            lntumorMutPriors = calcLnCombPrior(tumorMutGenes, dictGeneLength, v0)
            
        tumorMutGenes.append('A0')
        
        # collect data related to DEGs
        degGeneIndx = [i for i, j in enumerate(degMatrix.data[t,:]) if j == 1]
        tumorDEGGenes = [degGeneNames[i] for i in degGeneIndx]
        nTumorDEGs = len(degGeneIndx)  # corresponding to n, the number of DEGs in a given tumor
        tumorDEGMatrix = degMatrix.data[:,degGeneIndx]
        
        # calculate the pairwise likelihood that an SGA causes a DEG
        tumorLnFScore = calcF(tumorMutMatrix, tumorDEGMatrix,  alphaIJKList)        
        # Calculate the likelihood of expression data conditioning on A0, and then stack to 
        # the LnFScore, equivalent to adding a column of '1' to represent the A0 in tumorMutMatrix
        nullFscore = calcNullF(tumorDEGMatrix, alphaNull)
        tumorLnFScore = np.vstack((tumorLnFScore, nullFscore))  #check out this later
               
        # calcualte the prior probability that any of mutated genes can be a cause for a DEG,
        # tile it up to make an nTumorMutGenes x nTumorDEG matrix
        tumorMutPriorMatrix = np.tile(lntumorMutPriors, (nTumorDEGs, 1)).T
        
        lnFScore = add(tumorLnFScore, tumorMutPriorMatrix)
        
        # now we need to caclculate the normalized lnFScore so that each         
        columnAccumLogSum = np.zeros(nTumorDEGs)        
        for col in range(nTumorDEGs):
            currLogSum = np.NINF
            for j in range(lnFScore.shape[0]):
                if lnFScore[j,col] == np.NINF:
                    continue
                currLogSum = logSum(currLogSum, lnFScore[j,col])             
            columnAccumLogSum[col] = currLogSum
                
        normalizer = np.tile(columnAccumLogSum, (lnFScore.shape[0], 1))      

        posterior = np.exp(add(lnFScore, - normalizer))
        
        #write out the results        
        tumorPosterior = NamedMatrix(npMatrix = posterior, rownames = tumorMutGenes, colnames = tumorDEGGenes)
        if "\"" in tumorNames[t]:
            tumorNames[t] = tumorNames[t].replace("\"", "")    
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
# sum matrix ops

m1 = T.fmatrix()
m2 = T.fmatrix()
add = function([m1, m2], m1 + m2, allow_input_downcast=True)

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
 

def calcLnPrior(geneNames, dictGeneLength, v0):
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

    #extract gene lengths for all the genes in 'geneNames'
    listGeneLength = [dictGeneLength[g]  for g in geneNames]
    #Calculate the prior probability by taking each ###########FINISH THIS COMMENT
    inverseLength = [1 / float(x) for x in listGeneLength]
    sumInverseLength = sum(inverseLength)
    prior =  [(1-v0) * x / sumInverseLength for x in inverseLength] + [v0]
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
    prior =  [(1-v0)* x / sumInverseLength for x in inverseLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior 


#THIS FUNCTION NEEDS TESTING    
def calcLnCombORPrior(geneList, v0):
    ppiDict = readEdgeAllType_neighbors('BIOGRID-ORGANISM-Homo_sapiens-3.2.116.tab.txt')
    listGeneLength = []
    
    for gene in geneList:
        ppiNeighbors = ppiDict[gene]
        totalLengthofNeighbors = 0
        for n in ppiNeighbors:
            totalLengthofNeighbors += geneLengthDict[n]
        listGeneLength.append(totalLengthofNeighbors)

    inverseLength = [1 / float(x) for x in listGeneLength] 
    sumInverseLength = sum(inverseLength)
    prior =  [(1-v0)* x / sumInverseLength for x in inverseLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior

def main():
    
    geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

    mutMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GtM.csv"
    degMatrixFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/PANCAN.GeM.MaskedCNA.csv"
    outputFilePath = "/home/kevin/projects/TCIResults/Tumor.Type.Data/PANCAN/TestMP"

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
    calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, outputPath = outputFilePath,  dictGeneLength = geneLengthDict, rowBegin = 5, rowEnd = 10)#, opFlag = AND)
    


if __name__ == "__main__":
    main()       


   
    
  
   
   
 

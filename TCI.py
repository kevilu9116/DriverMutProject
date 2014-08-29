# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 22:03:45 2014

@author: kevin

This module contains functions required for performing the tumor-specific 
causal inference (TCI).  See Technical report "A method for identify driver
genomic alterations at the individual tumor level" by Cooper, Lu, Cai and Lu.

 
"""

import theano.tensor as T
from theano import function, shared,  config
import numpy as np
import sys
import math
from NamedMatrix import NamedMatrix
import scipy as s
from util import *

###############################################################################################
#Test Git Commit
"""
 The following block define the Theano functions that going to be used in the regular python funcitons
 """
"""
1.  Calculate ln(X + Y) based on ln(X) and ln(Y) using theano library

"""
    
maxExp = -4950.0 
x, y = T.fscalars(2)

yMinusx = y - x  ## this part is for the condition which x > y
xMinusy = x - y  # if x < y
bigger = T.switch(T.gt(x, y), x, y)
YSubtractX = T.switch(T.gt(x,y), yMinusx, xMinusy)       
 
#Then, when you want to use the sorted ones:    xy_sorted = T.sort(xy_init_vec)
x_prime =  T.log(1 + T.exp(YSubtractX)) + bigger
calcSum = T.switch(T.lt(YSubtractX, maxExp), bigger, x_prime)
logSum = function([x, y], calcSum, allow_input_downcast=True)
###################################################################

"""
2. sum matrix ops
"""
m1 = T.fmatrix()
m2 = T.fmatrix()

add = function([m1, m2], m1 + m2, allow_input_downcast=True)


    # declare a function that calcualte gammaln on a shared variable on GPU
aMatrix = shared(np.zeros((10, 10)), config.floatX)
gamma_ln = function([ ], T.gammaln(aMatrix))
theanoExp = function([ ], T.exp(aMatrix))
    
alpha = T.fscalar()
gamma_ln_scalar = function([alpha], T.gammaln(alpha), allow_input_downcast=True)

# now compute the second part of the F-score, which is the covariance of mut and deg
mutMatrix = shared(np.ones((10, 10)), config.floatX)  
expMatrix = shared(np.ones((10, 10)), config.floatX)
mDotE = function([], T.dot(mutMatrix, expMatrix))

ln_nMutMatrix_1 = shared(np.zeros((1000, 1000)), config.floatX)
ln_nMutMatrix_0 = shared(np.zeros((1000, 1000)), config.floatX) 
ln_nijk11 = shared(np.zeros((1000, 1000)), config.floatX)
ln_nijk10 = shared(np.zeros((1000, 1000)), config.floatX)  
ln_nijk01 = shared(np.zeros((1000, 1000)), config.floatX)
ln_nijk00 = shared(np.zeros((1000, 1000)), config.floatX)

fscore = function([], ln_nMutMatrix_1 + ln_nMutMatrix_0 + ln_nijk11 + ln_nijk10 +  ln_nijk01 + ln_nijk00 )

############################################################################################################



def calcTCI (mutcnaMatrixFN, degMatrixFN, alphaNull = [1, 1], alphaIJKList = [2, 1, 1, 2], v0=0.2, dictGeneLength = None, outputPath = "./Tumor.Type.Data/BLCA/testResults"):
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
    """
    
    # read in data in the form of NamedMatrix 
    try:
        mutcnaMatrix  = NamedMatrix(mutcnaMatrixFN)
    except:
        print "Failed to import data matrix %s\n", mutcnaMatrixFN
        sys.exit() 
        
    try:
        degMatrix = NamedMatrix(degMatrixFN)
    except:
        print "Failed to import data matrix %s\n", degMatrixFN
        sys.exit()
        
    if degMatrix.getRownames() != mutcnaMatrix.getRownames():
        print "The tumors for mutcnaMatrix and degMatrix do not fully overlap!"
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
    

    for t in range(nTumors):
        # collect data related to mutations
        tumormutGeneIndx = [i for i, j in enumerate(mutcnaMatrix.data[t,:]) if j == 1] #WHAT DOES THIS LINE DO?
        nTumorMutGenes = len(tumormutGeneIndx)
        tumorMutGenes=  [mutGeneNames[i] for i in tumormutGeneIndx]        
      
        #now extract the sub-matrix of mutcnaMatrix that only contain the genes that are mutated in a given tumor t
        # stack a column of '1' to represent the A0 
        tumorMutMatrix = mutcnaMatrix.data[:,  tumormutGeneIndx]
        
        ## check operation options:  1) orginal, do nothing and contiue
        # otherwise creat combinary matrix using the tumorMutMatrix 
        # createCombMatrix(tumorMutMatrix, operationFlag)
        
        lntumorMutPriors = calPrior(tumorMutGenes, dictGeneLength, v0)  # a m-dimension vector with m being number of mutations
        tumorMutGenes.append('A0')
        
        # collect data related to DEGs
        degGeneIndx = [i for i, j in enumerate(degMatrix.data[t,:]) if j == 1]
        tumorDEGGenes = [degGeneNames[i] for i in degGeneIndx]
        nTumorDEGs = len(degGeneIndx)  # corresponding to n, the number of DEGs in a given tumor
        tumorDEGMatrix = degMatrix.data[:,degGeneIndx]
        
        # calculate pair-wise m x n matrix
        tumorLnFScore = calcF(tumorMutMatrix, tumorDEGMatrix,  alphaIJKList)
        nullFscore = calcNullF(tumorDEGMatrix, alphaNull)
        tumorLnFScore = np.vstack((tumorLnFScore, nullFscore))  #check out this later
               
        # calcualte the prior probability that any of mutated genes can be a cause for a DEG,
        # tile it up to make an nTumorMutGenes x nTumorDEG matrix
        tumorMutPriorMatrix = np.tile(lntumorMutPriors, (nTumorDEGs, 1)).T
        
        lnFScore = add(tumorLnFScore, tumorMutPriorMatrix)
        
        tmp = NamedMatrix(npMatrix = tumorLnFScore, colnames = tumorDEGGenes, rownames = tumorMutGenes )
        tmp.writeToText(filePath, tumorNames[t]+ "-tumorLnFScore.csv")

        # now we need to caclculate the normalized lnFScore so that each         
        columnAccumLogSum = np.zeros(nTumorDEGs)        
        for col in range(nTumorDEGs):
            currLogSum = math.log(s.spacing(1))
            for j in range(lnFScore.shape[0]):
                currLogSum = logSum(currLogSum, lnFScore[j,col])             
            columnAccumLogSum[col] = currLogSum
                
        normalizer = np.tile(columnAccumLogSum, (lnFScore.shape[0], 1))      

        posterior = np.exp(add(lnFScore, - normalizer))
        
        #write out the results        
        tumorPosterior = NamedMatrix(npMatrix = posterior, rownames = tumorMutGenes, colnames = tumorDEGGenes)     
        tumorPosterior.writeToText(filePath, filename = tumorNames[t] + "-mut-vs-DEG-posterior.csv")
        

def calcNullF(degMatrix, alphaNull):
    """
    This funciton calculate the terms in equation #7 of white paper for 
    the leak cause node, which only require 3 terms because the cause exists 
    for every tumor.  A special prior is a special set of hyperparameter  
    
    """
    N = degMatrix.shape[0]
    # because all tumors have A0 set to 1
    term1 = s.special.gammaln (sum(alphaNull)) - s.special.gammaln(sum(alphaNull) + N) 
    term2 = map(s.special.gammaln, degMatrix.sum(axis = 0) + alphaNull[1]) - s.special.gammaln(alphaNull[1])
    term3 = map(s.special.gammaln, (degMatrix==0).sum(axis= 0) + alphaNull[0]) - s.special.gammaln(alphaNull[0])
    return  np.array(term1 + term2 + term3)
        
        
                            

def calcF(mutcnaMatrix, degMatrix, alphaIJKList):

    """
    This function calculate log funciton of the Eq 7 of TCI white paper 

    Input:     mutcnaMatrix      A N x m numpy matrix containing mutaiton and CNA data of N tumors and m genes
            degMatrix         A N x d numpy matrix containing DEGs from N tumors and d genes
               
                 alphaIJList     A list of two elements containing the hyperparameter define the prior distribution for mutation events
                 alphaIJKList    A list of four elements containing the hyperparameters defining the prior distribution of condition prability

    Output: A m x d matrix, in which each element contains the F-score of a pair of mutation and DEGs

    F-Score is calcaulated using the following equation.  \frac {\Gamma(\alpha_{ij})} {\Gamma(\alpha_{ij}+ N_{ij})}

    """
 
    # add check if mutcnaMatrix degMatrix is an instance of numpy float matrix of 32 bit
    if mutcnaMatrix.dtype != np.float32:
        mutcnaMatrix = mutcnaMatrix.astype(np.float32)
        #print "Data type for mutcnaMatrix not float32, downcasting matrix."
    if degMatrix.dtype != np.float32:
        degMatrix = degMatrix.astype(np.float32)
        #print "Data type for degMatrix not float32, downcasting matrix."
        
    # create 32bit theano copies of mutcan and DEG matrice, make them accessable to GPU
    mutcnaMatrix = shared(mutcnaMatrix.T, config.floatX)
    degMatrix =  shared(degMatrix, config.floatX)

    # calculate the first part of the F-scores
    ni1_vec = mutcnaMatrix.get_value().sum(axis = 1) + alphaIJKList[2] + alphaIJKList[3]  # a vector of length m contains total number cases in which m-th element are ONE
    ni0_vec = (mutcnaMatrix.get_value() == 0).sum(axis = 1) + alphaIJKList[0] + alphaIJKList[1] # a vector of length m contains total number of cases in which m-th element are ZERO
    
    # make a m x n matrix where a m-dimension vectior is copied n times
    aMatrix.set_value(np.tile(ni1_vec, (degMatrix.get_value().shape[1], 1)).T , config.floatX)
    ln_nMutMatrix_1.set_value(gamma_ln_scalar(alphaIJKList[2] + alphaIJKList[3]) -  gamma_ln(), config.floatX)
    
   
    aMatrix.set_value(np.tile(ni0_vec, (degMatrix.get_value().shape[1], 1)).T, config.floatX)    
    ln_nMutMatrix_0.set_value(gamma_ln_scalar(alphaIJKList[0] + alphaIJKList[1]) - gamma_ln(), config.floatX)


    #total number of cases in which mut == 1 and exp == 1
    mutMatrix.set_value(mutcnaMatrix.get_value(), config.floatX)
    expMatrix.set_value(degMatrix.get_value(), config.floatX)
    nijk_11 = shared (mDotE() + alphaIJKList[3], config.floatX) 

    # calc mut == 1 && deg == 0
    expMatrix.set_value(degMatrix.get_value() == 0, config.floatX)
    nijk_10 = shared(mDotE() + alphaIJKList[2], config.floatX)

    # calc mut == 0 && deg == 0
    mutMatrix.set_value(mutcnaMatrix.get_value() == 0, config.floatX)
    nijk_00 = shared(mDotE() + alphaIJKList[0], config.floatX)

    # calc mut == 0 && deg == 1
    expMatrix.set_value(degMatrix.get_value(), config.floatX)
    nijk_01 = shared(mDotE() + alphaIJKList[1], config.floatX)

    # calc the gammaln of the above matrix
    aMatrix.set_value(nijk_11.get_value(), config.floatX) 
    ln_nijk11.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[3]), config.floatX)
    
    aMatrix.set_value(nijk_10.get_value(), config.floatX)
    ln_nijk10.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[2]), config.floatX)

    aMatrix.set_value(nijk_01.get_value(), config.floatX)
    ln_nijk01.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[1]), config.floatX)

    aMatrix.set_value(nijk_00.get_value(), config.floatX)
    ln_nijk00.set_value(gamma_ln() - gamma_ln_scalar(alphaIJKList[0]), config.floatX)

    # now caluc the theano final
    fvalues = fscore()
    
    # check if the probability that mut == 1 && deg == 1 is bigger than mut == 0 && deg == 1, 
    # if yes, set the likelihood that mutated gene is a cause to zero
    condMutDEG_11 = nijk_11.get_value().T / ni1_vec   
    condMutDEG_01 = nijk_01.get_value().T / ni0_vec  
    elementsToSetZero = np.where( condMutDEG_11.T <= condMutDEG_01.T)
    fvalues[elementsToSetZero] = -100  # exponentiation of -17 is already zero
    
    return fvalues
 

def calPrior(geneNames, dictGeneLength, v0):
    """
    calPrior(geneNames, dictGeneLength, v0)
    
    Input: 
        geneNames       A list of SAG-affected genes that are altered in a give tumor
        dictGeneLength  A dictionary contain the length of all genes
        v0              A weight of a "leak node" besides SGA-affected genes 
                        that may contribute to the differential expression of a gene
                        
    """

    listGeneLength = [dictGeneLength[g]  for g in geneNames]
    inverseLength = [1 / float(x) for x in listGeneLength] 
    sumInverseLength = sum(inverseLength)
    prior =  [(1-v0)* x / sumInverseLength for x in inverseLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior 
 

    
def main():
    
    geneLengthDict = parseGeneLengthDict("Gene.Exome.Length.csv")
    calcTCI("/home/kevin/Dropbox (XinghuaLu)/TCI/Tumor.Type.Data/BLCA/BLCA.GtM.csv", "/home/kevin/Dropbox (XinghuaLu)/TCI/Tumor.Type.Data/BLCA/BLCA.GeM.csv", dictGeneLength = geneLengthDict)

if __name__ == "__main__":
    main()       


   
    
  
   
   
 

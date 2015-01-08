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
import numpy as np
import sys, os, math
import NamedMatrix
import scipy.special as ss
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

def calcTDI (mutcnaMatrixFN, degMatrixFN, alphaNull = [1, 1], alphaIJKList = [2, 1, 1, 2], 
              v0=0.3,   outputPath = ".", opFlag = None, rowBegin=0, rowEnd = None, GeGlobalDriverDict = None):
    """ 
    calcTCI (mutcnaMatrix, degMatrix, alphaIJList, alphaIJKList, dictGeneLength)
    
    Calculate the causal scores between each pair of SGA and DEG observed in each tumor
    
    Inputs:
        mutcnaMatrixFN      A file containing a N x G binary matrix containing the mutation and CNA 
                            data of all tumorss.  N is the number of tumors and 
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
        
    
        rowBegin, rowEnd    These two arguments control allow user to choose which block out of all tumors (defined by the two 
                            row numbers) will be processes in by this function.  This can be used to process
                            mulitple block in a parallel fashion.

        GeGlobalDriverDict  A dictionary, in which GEs are keys and values are top two GTs that have strongest global association with GEs


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
                         
    tumorNames = degMatrix.getRownames()
    nTumors, nMutGenes = mutcnaMatrix.shape()
    
    mutGeneNames = mutcnaMatrix.getColnames()
    degGeneNames = degMatrix.getColnames()
    
    # now we iterate through each tumor to infer the causal relationship between each 
    # pair of mut - deg
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
        
        
        # collect data related to DEGs.  Identify the genes that are differentially expressed in a tumor,
        # then collect
        degGeneIndx = [i for i, j in enumerate(degMatrix.data[t,:]) if j == 1]
        tumorDEGGenes = [degGeneNames[i] for i in degGeneIndx]
        nTumorDEGs = len(degGeneIndx)  # corresponding to n, the number of DEGs in a given tumor
        tumorDEGMatrix = degMatrix.data[:, degGeneIndx]
        
        # find the global drivers for the DEGs
        allTumorGlobDrivers = np.zeros((nTumors, nTumorDEGs))
        gDriverGeneNames = []
        gDriverIndices = []
        g2ndDriverGeneNames = []
        missingGDrivers = []
        for  j, ge in enumerate(tumorDEGGenes):
            if ge not in GeGlobalDriverDict:
                ge = ge.replace("-", ".")
                if ge not in GeGlobalDriverDict: # if still cannot find it, rel
                    missingGDrivers.append(j)
                    gDriverIndices.append(0)
                    continue
            top2Driver4CurGE = GeGlobalDriverDict[ge] # query the dictionary to retrive the top two global dirvers
            geGlobalDriver = top2Driver4CurGE[0]  
            gDriverGeneNames.append(geGlobalDriver)
            g2ndDriverGeneNames.append(top2Driver4CurGE[1])
            gDriverIndices.append(mutGeneNames.index(geGlobalDriver))
        allTumorGlobDrivers = mutcnaMatrix.data [:, gDriverIndices]
        for m in missingGDrivers:
            allTumorGlobDrivers[:, m] = 1

        # collect data related to mutations
        tumormutGeneIndx = [i for i, j in enumerate(mutcnaMatrix.data[t,:]) if j == 1]
        tumorMutGenes = [mutGeneNames[i] for i in tumormutGeneIndx] 
        print "Processing tumor " + tumorNames[t] + " with " + str(nTumorDEGs) + " DEGs, " + str(len(tumormutGeneIndx)) + " GTs"  
        # the matrix that cotnains the likelihood of pair wise causal structure, rows are GTs; columns are GEs
        tumorLnFScore = np.zeros((len(tumorMutGenes), nTumorDEGs))

        # We need to loop through individual GTs, divide tumors into blocks: one with a GT is set to ones, the other in which GT is 0,
        # then calculate the using the GT can explain the DEGs in a tumor well, while enhance the capbility of and other 
        # global driver to explain the GE in the tumors that do not have the mutation in the GT of interest  
        for i, gtName in enumerate(tumorMutGenes):
            gtIndx = tumormutGeneIndx[i]
            #print "Index of current GT: ", str(gtIndx) + " and name: " + gtName
            indxTumorWithGT = np.where(mutcnaMatrix.data[:, gtIndx] == 1)[0]
            indxTumorNotGT = np.where(mutcnaMatrix.data[:,gtIndx] == 0)[0] 

            # first calculate the FScore for tumors with T.  We can use the CalNullF which deal with all GT == 1 cases
            DEGMatrixOfTumorsWithGT = tumorDEGMatrix[indxTumorWithGT,:]
            tumorLnFScore[i,:] = calcNullF(DEGMatrixOfTumorsWithGT, [1, 2])

            # now dealing with tumors without GT perturbed
            DEGMatrixOfTumorsNoGT = tumorDEGMatrix[indxTumorNotGT,:]
            # extract the global drivers for the DEGs
            globalDriver4DEGs = allTumorGlobDrivers[indxTumorNotGT,:]
            if gtName in gDriverGeneNames:  # swap global driver if the cur GT is one
                indxOfColumnToSwap = gDriverGeneNames.index(gtName)
                indxOfReplaceGDriver = mutGeneNames.index(g2ndDriverGeneNames[indxOfColumnToSwap])
                globalDriver4DEGs[:,indxOfColumnToSwap] = mutcnaMatrix.data [indxTumorNotGT, indxOfReplaceGDriver]
                
            # call calcF, which should return a vector gloabal drivers to GEs,
            # Note that we do not need to calcualte the prior for global drivers because they are the same for a
            # given GE, therefore will be cancelled out when normalize across GT
            tumorLnFScore[i, :] = tumorLnFScore[i, :] + calcF(globalDriver4DEGs, DEGMatrixOfTumorsNoGT, alphaIJKList)
        # Done with data likelihood p(D|S) section

       # Calculate the likelihood of expression data conditioning on A0, and then stack to 
        # the LnFScore, equivalent to adding a column of '1' to represent the A0 in tumorMutMatrix
        nullFscore = calcNullF(tumorDEGMatrix, alphaNull)
        tumorLnFScore = np.vstack((tumorLnFScore, nullFscore))  
               
        # calcualte the prior probability that any of mutated genes can be a cause for a DEG,
        # tile it up to make an nTumorMutGenes x nTumorDEG matrix
        lntumorMutPriors = calcLnPriorBasedOnFreqWithV0(tumormutGeneIndx, mutcnaMatrix, v0)
        lnFScore = np.add(tumorLnFScore.T, lntumorMutPriors).T
        
        # now we need to caclculate the normalized lnFScore so that each         
        columnAccumLogSum = calcColNormalizer (lnFScore) 
        posterior = np.exp(np.add(lnFScore, - columnAccumLogSum))
        
        #write out the results                       
        tumorMutGenes.append('A0') 
        tumorPosterior = NamedMatrix(npMatrix = posterior, rownames = tumorMutGenes, colnames = tumorDEGGenes)
        if "\"" in tumorNames[t]:
            tumorNames[t] = tumorNames[t].replace("\"", "")    
        tumorPosterior.writeToText(filePath = outputPath, filename = tumorNames[t] + ".csv")

        
def calcF(mutcnaInputMatrix, degInputMatrix, alphaIJKList):
    """
    This function calculate log funciton of the Eq 7 of TCI white paper.  Here we deal with a special case, in that 
    we aim to calculate the likelihood of a DEG to be explained by the best global drivr. 
    
    Input:  mutcnaInputMatrix      A N x m numpy matrix containing m SGAs.  Each colmn contains the state of the most likely
                                   of the corresponding DEG in the next matrix.   
            degInputMatrix         A N x m numpy matrix containing DEGs from N tumors and m DEGs 
               
            alphaIJKList        A list of Dirichlet hyperparameters for caulate the prior
                            of condition probability parameters. alphaIJK[0]: mut == 0 && deg == 0;
                            alphaIJK[1]: mut == 0 && deg == 1; alphaIJK[2]: mut == 1 && deg == 0;
                            alphaIJK[3]: mut == 1 && deg == 1
             alphaIJKList    A list of four elements containing the hyperparameters defining the prior distribution of condition prability
    
    Output: A vector of Fscore of a pair of GT and GE 
    
    F-Score is calcaulated using the following equation.  \frac {\Gamma(\alpha_{ij})} {\Gamma(\alpha_{ij}+ N_{ij})}
    """
    
    if mutcnaInputMatrix.shape != degInputMatrix.shape:
        print "The shape of mutcnaInputMatrix and degInputMatrix: " + str(mutcnaInputMatrix.shape) + " : " + str(degInputMatrix.shape)
        raise Exception("Try to call \"calcF\" with two matrices with non-matching shape")

    # calculate the first part of the F-scores, which collect total counts of Gt across tumors
    ni0_vec = np.sum(mutcnaInputMatrix== 0, axis = 0) + alphaIJKList[0] + alphaIJKList[1] # a vector of length m contains total number of cases in which m-th element are ZERO
    ni1_vec = np.sum (mutcnaInputMatrix, axis = 0 ) + alphaIJKList[2] + alphaIJKList[3]  # a vector of length m contains total number cases in which m-th element are ONE

    glnNi0_vec = ss.gammaln(alphaIJKList[0] + alphaIJKList[1]) - ss.gammaln(ni0_vec)
    glnNi1_vec = ss.gammaln(alphaIJKList[2] + alphaIJKList[3]) - ss.gammaln(ni1_vec)

     #Initialize fscore matrix to zero
    nTumors, nDEGs = degInputMatrix.shape

    # add the first term of the Equation 7, two gammaln terms associated with the p(GT == 0) and p(GT == 1)
    fscore = glnNi0_vec + glnNi1_vec
    
    # possible combination of GT and GE
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]  
    for i in range(nDEGs):
        tmpFScore = 0
        for alpha, state in enumerate(states):
            Nijk = np.sum((mutcnaInputMatrix[:, i] == state[0]) * (degInputMatrix[:, i] == state[1]))
            tmpFScore = tmpFScore + ss.gammaln(Nijk + alphaIJKList[alpha]) - ss.gammaln(alphaIJKList[alpha]) 

        fscore[i] =  fscore[i] + tmpFScore
    return fscore
 
def main():
    
    geneLengthDict = parseGeneLengthDict("/home/kevin/projects/TCIResults/Tumor.Type.Data/Gene.Exome.Length.csv")

    mutMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/chunhui.testmatrices/GtM.testset.csv"
    degMatrixFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/chunhui.testmatrices/GeM.testset.csv"
    outputFilePath = "/home/kevin/GroupDropbox/Chunhui Collab/Tumor.Type.Data/chunhui.testmatrices/TCIORTestResults"

    calcTCI(mutcnaMatrixFN=mutMatrixFilePath, degMatrixFN=degMatrixFilePath, outputPath = outputFilePath,  dictGeneLength = geneLengthDict,
            opFlag = OR, ppiFile = "BIOGRID-ORGANISM-Homo_sapiens-3.2.116.tab.txt")#, opFlag = AND)
    


if __name__ == "__main__":
    main()       


   
    
  
   
   
 

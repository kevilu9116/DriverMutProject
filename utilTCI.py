"""
Created on Tue Aug 26 16:33:00 2014

@author: kevin Lu
"""
import numpy as np
from NamedMatrix import NamedMatrix
import theano.tensor as T
from theano import function, shared,  config
import sys
import math
import os
from NamedMatrix import NamedMatrix
import scipy as s
import scipy.sparse as sp
import csv, time, random

def parseGeneLengthDict(fileName):
    inputFile = open(fileName, "r")
    geneDict = {}
    inputFile.readline() #read over first line
    
    #populate geneDict
    #keys: gene name
    #value: gene length
    for line in inputFile:
        curLine = line.strip().split(",")
        geneDict[curLine[0]] = curLine[1]
    
    inputFile.close()
    return geneDict

# define constant represent different combinaiton of mutation data
AND = 1
OR = 2  
    
def createANDComboMatrix(tumorMutGenes, mutcnaMatrix):
    """
    createComb(mutationMatrix, flag):

    Input:
        mutationMatrix      An m x n matrix consisting of m tumor mutations and n potentially altered genes.
        flag                A flag keyword that is set to either "AND" or "OR" that specifies the type of logical relation
                            observed between two distinct genes.

    Output:                 An m x ((n-1) * n) / 2 matrix consisting of m tumors and ((n-1) * n) / 2 gene pair combinations

    """

    #get necessary data to create the dimensions of our final output matrix
    #(list of genes, dimensions of input mutation matrix, new number of columns for the output matrix)
    geneList = mutationMatrix.getColnames()
    numRows, numCols = np.shape(mutationMatrix.data)
    if numCols > 500:
        print "Too many combinations. Skip"
        return None
    newNumCols = ((numCols - 1) * numCols) / 2
    tmpColNames = []

    outputMatrix = np.zeros((numRows, newNumCols), dtype = np.float32)   
    
    #iterate through our input matrix and generate every non-repeating permutations of 2 distinct genes.
    #For each pair, create the name "Gene1/Gene2" for that pair, then for each tumor, do an "AND" or "OR"
    #operation between the two values
    count = 0
    for i in range(len(geneList) - 1):
        for j in range(i + 1, len(geneList)):
            gene1Vals = mutationMatrix.data[:, i]
            gene2Vals = mutationMatrix.data[:, j]
            tmpColNames.append(geneList[i] + "/" + geneList[j])
            #'AND' the two values together
            if flag == AND:
                outputMatrix[:, count] = gene1Vals * gene2Vals
            #'OR the two values together'
            else:
                print "Flag operation was not defined. Please specify \"AND\" or \"OR\" as your flag."
                sys.exit()
            count += 1
    
    # clean the columns that have too few 1s based on a 2% threshold
    totalOnes = outputMatrix.sum(axis = 0)
    colsToKeep = np.where((totalOnes / numRows) > .02)[0]
    if colsToKeep.size == 0:
        print "None of the gene combinations meet the required frequency of .02 in the population."
        return None
    elif colsToKeep.size > 10000:
        print "Number of gene combinations exceeds 10000. Skipping over."
        return None
    outputMatrix = outputMatrix[:, colsToKeep]
    newColNames = [tmpColNames[colsToKeep[i]] for i in range(colsToKeep.size)]

    return NamedMatrix(npMatrix = outputMatrix, colnames = newColNames, rownames = mutationMatrix.getRownames())

## 
def createORComb(geneList, ppiDict, mutCNAMatrix):
    """

    """
    tumorMutMatrix = mutCNAMatrix.getValuesByCol(geneList) #this matrix does not contain the gene names
    allMutGeneNames = mutCNAMatrix.getColnames()

    for i, g in enumerate(geneList):
        #ppiNeighbors = ppiDict[g]
        ppiNeighbors = ppiDict[g].keys()
        if len(ppiNeighbors) == 0:
            continue
        ppiNeighbors = set(i.upper() for i in ppiNeighbors)
        ppiNeighbors = ppiNeighbors.intersection(set(allMutGeneNames))
        ppiNeighbors = list(ppiNeighbors)        
        if len(ppiNeighbors) == 0:
            continue
        
        neighborSubMatrix = mutCNAMatrix.getValuesByCol(ppiNeighbors)
        # identify the rows in which at least one colmn contains a one
        neighborColSum  = np.sum(neighborSubMatrix, 1) 
        tumorMutMatrix[np.where(neighborColSum > 0)[0],i] = 1
        
    return tumorMutMatrix


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
    
 

def calcLnCombANDPrior(combGeneNames, geneLengthDict, v0):
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


def calcPanCanLnCombANDPrior(combGeneNames, geneLengthDict, vtprior, v0):
    raise Exception ("calcPanCanLnCombANDPrior not implemented")
 

#THIS FUNCTION NEEDS TESTING    
def calcLnCombORPrior(geneList, ppiDict, geneLengthDict, fullMutGeneNames, v0):
    listGeneLength = []
    
    for gene in geneList:
        ppiNeighbors = ppiDict[gene].keys()
        # change all to upper cases and remove duplicates 
        ppiNeighbors = set(i.upper() for i in ppiNeighbors)
        ppiNeighbors = ppiNeighbors.intersection(set(fullMutGeneNames))
        ppiNeighbors = list(ppiNeighbors)
        if len(ppiNeighbors) == 0:
            listGeneLength.append(geneLengthDict[gene])
            continue
        
        totalLengthofNeighbors = 0
        for n in ppiNeighbors:
            totalLengthofNeighbors += geneLengthDict[n]
        listGeneLength.append(totalLengthofNeighbors)

    inverseLength = [1 / float(x) for x in listGeneLength] 
    sumInverseLength = sum(inverseLength)
    prior =  [(1-v0) * x / sumInverseLength for x in inverseLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior

   

def calcPanCanLnCombORPrior(geneList, ppiDict, geneLengthDict, fullMutGeneNames, vtprior, v0):
    listGeneLength = []
    
    for gene in geneList:
        ppiNeighbors = ppiDict[gene].keys()
        # change all to upper cases and remove duplicates 
        ppiNeighbors = set(i.upper() for i in ppiNeighbors)
        ppiNeighbors = ppiNeighbors.intersection(set(fullMutGeneNames))
        ppiNeighbors = list(ppiNeighbors)
        if len(ppiNeighbors) == 0:
            listGeneLength.append(geneLengthDict[gene])
            continue
        
        totalLengthofNeighbors = 0
        for n in ppiNeighbors:
            totalLengthofNeighbors += geneLengthDict[n]
        listGeneLength.append(totalLengthofNeighbors)

    inverseGeneLength = [1 / float(x) for x in listGeneLength] 
    inverseGeneLength = [float(x) * vtprior for x in inverseGeneLength[:-1]] + inverseGeneLength 
    sumInverseLength = sum(inverseGeneLength)
    prior =  [(1-v0) * x / sumInverseLength for x in inverseGeneLength] + [v0]
    lnprior = [math.log(x) for x in prior]
    return lnprior
    
    
    
## This function calculate the logsum of each columns of a matrix.
def calcColNormalizer(inMatrix):
    #Theano function for calculating logSum, i.e., calculate ln(X + Y) based on ln(X) and ln(Y).
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
    
    # now we  caclculate sum of log joint as normalizer 
    if len(inMatrix.shape) < 2:
        raise Exception ("calcColNormalizer expect a 2D matrix")
    nRows, nCols = inMatrix.shape        
    columnAccumLogSum = np.zeros(nCols)        
    for col in range(nCols):
        currLogSum = np.NINF
        for j in range(nRows):
            if inMatrix[j,col] == np.NINF:
                continue
            currLogSum = logSum(currLogSum, inMatrix[j,col])             
        columnAccumLogSum[col] = currLogSum
        
    return columnAccumLogSum

"""
"""
def readEdgeAllType_neighbors(P_tfFile):
   L_rets = {}
   L_rets2 = {}
   L_neighbors = {}
   countTotalLine = 0
   L_TFname = open(P_tfFile, 'r')

   count=0
   for TFtp in L_TFname:
      count += 1
      if count>36:
         countTotalLine += 1
         compTp = TFtp.split("\t")
         if len(compTp[2])>0 and len(compTp[3])>0:
            v1 = compTp[2]; v2 = compTp[3]
            if v1 != v2:
               edge = [v1,v2]
               edge.sort()
               keyTp = edge[0]+'|'+edge[1]
               L_rets[keyTp] = 1
               L_rets2[v1]=0;L_rets2[v2]=0
               if not L_neighbors.has_key(v1):
                  L_neighbors[v1] = {}
               L_neighbors[v1][v2] = 1
               if not L_neighbors.has_key(v2):
                  L_neighbors[v2] = {}
               L_neighbors[v2][v1] = 1

   L_nodes = []
   for item in L_rets2:
      L_nodes.append(item)
   L_nodes.sort()
   print 'BIOGRID: PPI number:',len(L_rets), ' Protein number:',len(L_rets2)
   return L_neighbors


 


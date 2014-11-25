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
AND_PLUS_ORIGINAL = 2  
    
def createANDComb(mutationMatrix, flag = AND):
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

def createORComb(geneList, mutCNAMatrix):
    ppiDict = readEdgeAllType_neighbors('BIOGRID-ORGANISM-Homo_sapiens-3.2.116.tab.txt')
    tumorMutMatrix = mutCNAMatrix.getValuesByCol(geneList) #this matrix does not contain the gene names
    allMutGeneNames = mutCNAMatrix.getColnames()

    for i, g in enumerate(geneList):
        ppiNeighbors = ppiDict[g]
        ppiNeighbors = set(ppiNeighbors).intersection(set(allMutGeneNames))
        neighborSubMatrix = mutCNAMatrix.getValuesByCol(ppiNeighbors)
        curGeneGT = tumorMutMatrix[:,i]
        for j in range(len(neighborSubMatrix[0])):
            curCol = neighborSubMatrix[:, j]
            curGeneGT[np.where(curCol == 1)[0]] = 1
        tumorMutMatrix[:, i] = curGeneGT   

"""
 The following block define the Theano functions that going to be used in the regular python funcitons
 """
"""
1.  Calculate ln(X + Y) based on ln(X) and ln(Y) using theano library

"""
## This function calculate the logsum of each columns of a matrix.
def calcColNormalizer(inMatrix):
    #Theano function for calculating logSum
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


 


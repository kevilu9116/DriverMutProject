"""
Created on Tue Aug 26 16:33:00 2014

@author: kevin
"""

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
    
    

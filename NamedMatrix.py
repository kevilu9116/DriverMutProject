# -*- coding: utf-8 -*-
"""

NamedMatrix (filename = None, delimiter = ',', npMatrix = None, colnames = None, rownames = None))

A wrapper class enable access data matrix elements by col and row names.

This class is capable of parsing csv files and keep track the row and column names.
Elements such as rows and columns or a block of matrix according to names of row and column
as well as index arrays

Created on Sun Aug 25 08:40:33 2013



@author: xinghualu
"""

import numpy as np
import sys
from StringIO import StringIO

class NamedMatrix:
    ## Constructor
    #  @param  filename=None  A string point to a text matrix file
    #  @param delimiter=','  A string indicate the delimiter separating fields in txt
    #  @param npMatrix=None  A reference to a numpy matrix
    #  @colnames  A string array of column names
    #  @rownames  A string array of rownames

    def __init__(self, filename = None, delimiter = ',', npMatrix = None, colnames = None, rownames = None):
        
        if filename and npMatrix:  
            raise Exception ("Cannot create a NamedMatrix with both 'npMatrix' and 'filename' arguments set")
        if not filename and  npMatrix == None:
            raise Exception ("Attempt to create a NameMatrix without 'filename' or an 'npMatrix'")
        
        if filename:
            print "Extracting matrix file " + filename
            try:
                f = open(filename, 'r')
                lines = f.readlines()
                f.close()
            except IOError:
                print "Fail to read  file " + filename
                return
            
            if len(lines) == 1:  # Mac version csv, end with "\r" instead of "\n" as return
                lines = lines[0].split("\r")
                self.colnames = lines.pop(0).strip().split(',') # split  header and extract colnames
                map(lambda x: x.strip(), lines)  # remove the "\r"
                lines = "\n".join(lines)  # use "\n" to join lines
            else:
                self.colnames = lines.pop(0).strip().split(',')
                self.colnames.pop(0)
                self.colnames = [x.translate(None, '"') for x in self.colnames]

                
                lines = "".join(lines)
                
            # extract condition name
            self.rownames = list()            
            for l in lines.split("\n"):
                self.rownames.append(l.split(',')[0]) 
            
            if lines[-1]== '\n' :
                self.rownames.pop()
                            
            # read in data and generate a numpy data matrix
            self.data = np.genfromtxt(StringIO(lines), delimiter = ",", usecols=tuple(range(1, len(self.colnames)+1)))
            if self.data.shape[0] != len(self.rownames): 
                print "Name matrix: When parsing %s, The size of matrix does not match the length of  rownames" %filename
                print self.rownames
                print self.data.shape
                raise Exception ()
                
            if  self.data.shape[1] != len(self.colnames):
                print "Name matrix: When parsing %s, The size of matrix does not match the length of  colnames" %filename
                raise Exception()
                
            
        if npMatrix != None:
            self.data = npMatrix
            nrow, ncol = np.shape(self.data)
            if colnames:
                if len(colnames) == ncol:
                    self.colnames = colnames
                else:
                    raise Exception("Dimensions of input colnames and matrix do not agree")
            else:
                self.colnames = list()
                for c in range(ncol):
                    self.colnames.append('c' + str(c))
            if rownames:
                if len(rownames) == nrow:
                    self.rownames = rownames
                else:
                    raise Exception("Dimensions of input rownames and matrix do not agree")
            else:
                self.rownames = list()
                for r in range(nrow):
                    self.rownames.append('r' + str(r))
                    
        self.nrows, self.ncols = np.shape(self.data)
        
                    
    def setColnames(self, colnames):
        ## set the column names 
    
        if len(colnames) == len(self.colnames):
            self.colnames = colnames
        elif len(colnames) == self.data.shape[1]:
            self.colnames = colnames
        else:
            raise Exception("New colnames vector has differnt dimension as the original colnames")
            
    def getColnames(self):
        return self.colnames
            
    def setRownames(self, rownames):
        if len(rownames) == len(self.rownames):
            self.rownames = rownames
        elif len(rownames) == self.data.shape[0]:
            self.rownames = rownames
        else:
            raise Exception("New rownames vector has differnt dimension as the original colnames")
    
    def getRownames(self):
        return self.rownames
            
    def getValuesByCol(self, colnames):
        if isinstance (colnames, list):
            if not set(colnames) <= set(self.colnames):
                raise Exception("Try to access nonexisting columns")
            else:
                colIndx = map(lambda x: self.colnames.index(x), colnames)
                ixgrid = np.ix_(range(self.nrows), colIndx)
                return self.data[ixgrid]

        if isinstance(colnames, basestring): 
            if colnames not in self.colnames:
                raise Exception ("Try to access non-existing column")
            else:
                return self.data[:, self.colnames.index(colnames)]
                
        
    def setValuesByColName(self, values, col):      
        self.data[:,self.colnames.index(col)] = values
        
        
     
    def shape(self):
        if self.data != None:
            return np.shape(self.data)
            
        else:
            return None
            
    ## Return the position indices of colnames  
    def findColIndices(self, colnames):
        if isinstance (colnames, list):
            if not set(colnames) <= set(self.colnames):
                raise Exception("Try to access nonexisting columns")
            else:
                colIndx = map(lambda x: self.colnames.index(x), colnames)
                return colIndx

        if isinstance(colnames, basestring): 
            if colnames not in self.colnames:
                raise Exception ("Try to access non-existing column")
            else:
                return self.colnames.index(colnames)
                
        
    ## Return the position indices of rownames 
    def findRowIndices(self, rownames):
        if set(rownames) - set(self.rownames):
            raise Exception("Unknown column name is used to query index")
            
        return [lambda x: self.rownames.index(x) for x in rownames]
        
    
    def setCellValue(self, rowname, colname, value):
        value = np.float(value) # force it into a np.float
        self.data[self.rownames.index(rowname), self.colnames.index(colname)] = value
        
    
    ## Output matrix to a text file
    def writeToText(self, filePath, filename, delimiter=','):
        try:
            outMatrix = open(filePath + "/" + filename, "w") #Open file containing the output matrix
        except:
            print "Could not find filepath to output file. Please ensure you have given an existing filepath."
            sys.exit()

        #Writing out the column header. Iterate through colnames in our class, and write
        # them out to the first line of the file, seperated by the given delimiter.

        outMatrix.write("Sample") #Write "Sample" as the first cell of our matrix. Properly aligns the rows and columns.
        for colName in self.colnames: #Write out rest of column headers
            outMatrix.write(delimiter + colName)
        outMatrix.write("\n")

        #Write out each row of our matrix
        for i in range(self.shape()[0]):
            outMatrix.write(self.rownames[i]) #Write out the rowName for the particular row
            for j in self.data[i]:
                outMatrix.write(delimiter + str(j)) #Write out each cell of data for that row, separated by the given delimiter
            outMatrix.write("\n")
            
        outMatrix.close() #Done writing matrix, close file
    
    
        
            
        
    
    
    
        
        
            
            
            
            
                        
  
                    

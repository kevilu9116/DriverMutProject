import os, re, sys
from NamedMatrix import *
import numpy as np

GeMatricesFolder =  "/home/xinghua/projects/TCIResults/Tumor.Type.Data/"
GtMutInfoFolder = "/home/xinghua/Dropbox/TCI/GtM.withmutinfo/"
mutInfoFiles = os.listdir(GtMutInfoFolder)

regexprs = re.compile("(\w+?)\.GtM\.withmutinfo\.csv")

for fn in mutInfoFiles:
    if regexprs.match(fn):
        cancerType  = fn.split('.')[0]
    else: 
        continue
    
    
    mutInfoFile = GtMutInfoFolder + fn
    
    print "Loading " + mutInfoFile
    mutInfoMatrix = NamedMatrix(mutInfoFile)
    if not mutInfoMatrix:
        print "Fail to load " + mutInfoFile
        sys.exit()
    
    GeMfn = GeMatricesFolder  + cancerType + "/" + cancerType + ".GeM.csv"
    print "Loading " + GeMfn
    GeMatrix = NamedMatrix(GeMfn)
    if not GeMatrix:
        print "Fail to load " + GeMfn
        sys.exit()
     
    
    if mutInfoMatrix.getRownames() != GeMatrix.getRownames():
        print "Rownames of mutInfoMatrix do not agree with those of GeM"
        sys.exit()
        
    nTumors = GeMatrix.shape()[0]
    for i in range(nTumors):
        # find indices of cells with SCNA
        scnaCells = np.where(mutInfoMatrix.data[i,] < 0)[0]  # cells with SCNA deletions
        np.append(scnaCells, np.where(mutInfoMatrix.data[i,] > 1)[0])
        
        GeMatrix.data[i, scnaCells] = 0
        
    GeMatrix.writeToText(filename = GeMfn)  
        
        
    
    
    
    
        
        
        
    


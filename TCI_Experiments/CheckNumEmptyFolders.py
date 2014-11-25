# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:14:54 2014

@author: xinghualu
"""
import os
folderList = os.listdir("/home/kevin/GroupDropbox/TCI/Tumor.Type.Data")
for cancer in folderList:
    if not "." in cancer:
        cancerFiles = os.listdir("/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer)
        mutMatrixFilePath = "null"
        degMatrixFilePath = "null"
        outputFilePath = "/home/kevin/GroupDropbox/TCI/Tumor.Type.Data/" + cancer + "/CombLogicAND.Results"
        if os.listdir(outputFilePath) == []:
            print cancer


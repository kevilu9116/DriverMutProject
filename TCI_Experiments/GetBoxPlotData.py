import sys
import os
import numpy as np
from NamedMatrix import NamedMatrix

filePath = sys.argv[1]
myFiles = os.listdir(filePath)
for f in myFiles:
	if (".GtM" in f or ".GeM" in f) and ".~lock" not in f:
		myInputMatrix = NamedMatrix(filePath + "/" + f)

		#Get rowsums of tumor, the obtain the mean and standard deviation of the rowsums
		rowSum = myInputMatrix.data.sum(axis=1)
		
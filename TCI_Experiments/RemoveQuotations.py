import os
import sys

myDir = sys.argv[1]
for filename in os.listdir(myDir):
	newName = filename.replace("\"", "")
	os.rename(myDir + "/" + filename, myDir + "/" + newName)
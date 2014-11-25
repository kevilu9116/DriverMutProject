read.csvfile <- function(fname) {
  data.matrix = as.matrix(read.csv(fname, header=F))
  rowids = data.matrix[-1,1]
  colids = data.matrix[1,-1]
  data.matrix = data.matrix[-1,]
  data.matrix = data.matrix[,-1]
  data.matrix = matrix(as.numeric(data.matrix), dim(data.matrix))
  rownames(data.matrix) = rowids
  colnames(data.matrix) = colids
  
  return(data.matrix)
}

#set working directory and read PANCAN file
setwd("~/projects/TCIResults/Tumor.Type.Data/PANCAN/")
pancanMatrix = read.csvfile("PANCAN.GtM.csv")
#Get a vector representing the frequency % for each gene (coverage %)
geneColSum = colSums(pancanMatrix)
geneFrequencyVector = testColSums / nrow(pancanMatrix)

maxGTVec  = numeric(length = nrow(pancanMatrix))
for (r in 1:nrow(pancanMatrix))
{
  
  curTumorOnes = which(pancanMatrix[r,] == 1)
  if(length(curTumorOnes) == 0)
  {
    next
  }
  maxValue = max(geneFrequencyVector[curTumorOnes])
  maxGTVec[r] = maxValue
}

hist(maxGTVec)

# Read GeM matrix
# pancanGEmatrix = read.csvfile("PANCAN.GeM.csv")

# Create new GT and GE matrix based on tumors to keep
# numGTsPerTumor = rowSums(pancanMatrix)
# tumors.2keep = names(which(numGTsPerTumor>=3))
# tumors.2delete = names(which(numGTsPerTumor < 3))

# pancanGEmatrix.new = pancanGEmatrix[tumors.2keep, ]
# pancanGTmatrix.new = pancanMatrix[tumors.2keep, ]
# 
# dim(pancanGEmatrix.new)
# 
# write.csv(pancanGEmatrix.new, file="PANCAN.GeM.numGTslargerthan2.csv")
# write.csv(pancanGTmatrix.new, file="PANCAN.GtM.numGTslargerthan2.csv")
# 
# write.csv(tumors.2delete, file="~/GroupDropbox/Chunhui Collab/Tumors.to.Delete.csv")



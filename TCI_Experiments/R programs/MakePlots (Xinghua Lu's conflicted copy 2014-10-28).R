### task 1
numofmutations.percancer = list()

# forloop start
setwd("~/GroupDropbox/TCI/Tumor.Type.Data")
tumorTypeFolders = list.files()

for(i in 1:length(tumorTypeFolders))
{
  if(grepl(".", tumorTypeFolders[i], fixed = TRUE))
  {
    print(paste("Skip over file", tumorTypeFolders[i]))
    next
  }

  
  data = read.csv(paste(tumorTypeFolders[i], "/", tumorTypeFolders[i], ".GtM.csv", sep = ""), header=FALSE)
  data.rownames = as.matrix(data[-1,1])
  data.colnames = as.matrix(data[1,-1])
  data.matrix = data[-1,]
  data.matrix = data.matrix[,-1]
  data.matrix = matrix(as.numeric(as.matrix(data.matrix)), dim(data.matrix))
  mutation.per.tumor = rowSums(data.matrix, na.rm=TRUE)
  numofmutations.percancer[[tumorTypeFolders[i]]] = mutation.per.tumor
  
}

boxplot(numofmutations.percancer)
#cancertype.i = "BLCA"
#data = read.csv("BLCA/BLCA.GtM.csv", header=FALSE)


# calc rowSum


# forloop end

### task 2
require(gplots)

data.input.matrix = read.csv("PANCAN/subMatFinal.csv", header=FALSE)
tumorids = data.input.matrix[-1,1]
top50muts = data.input.matrix[1,-1]
mutation.matrix = data.input.matrix[-1,]
mutation.matrix = mutation.matrix[,-1]
mutation.matrix = matrix(as.numeric(as.matrix(mutation.matrix)), dim(mutation.matrix))
indx.nomuts = which(rowSums(mutation.matrix)==0)
mutation.matrix = mutation.matrix[-indx.nomuts,]

heatmap.plot = heatmap.2(mutation.matrix)





heatmap.plot = heatmap.2(data.submatrix)


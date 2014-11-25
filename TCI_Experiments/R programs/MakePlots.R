### task 1
numofmutations.percancer = list()

# forloop start
setwd("~/projects/TCIResults/Tumor.Type.Data/")
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

numofmutations.percancer.noPANCAN = numofmutations.percancer[-which(names(numofmutations.percancer)=="PANCAN")]
fname = "MutationFreq.vs.CancerType.BWboxplot.jpeg"
jpeg(fname, res=300, width=2000, height=1200)
boxplot(numofmutations.percancer, outline=F, ylab="Mutation Frequency", ylim=c(0, 800), cex.axis=0.5, cex.lab=0.5, las=1)
dev.off()

#calc rowSum


#forloop end

## task 2
require(gplots)

# Load data
data.input.matrix = read.csv("finalTop100GtM.csv", header=FALSE)
tumorids = data.input.matrix[-1,1]
top50muts = data.input.matrix[1,-1]
mutation.matrix = data.input.matrix[-1,]
mutation.matrix = mutation.matrix[,-1]
mutation.matrix = matrix(as.numeric(as.matrix(mutation.matrix)), dim(mutation.matrix))
indx.nomuts = which(rowSums(mutation.matrix)==0)
mutation.matrix = mutation.matrix[-indx.nomuts,]


mutation.matrix.100tumorsamples = mutation.matrix[1:100,]
rownames(mutation.matrix.100tumorsamples) = tumorids[1:100]
colnames(mutation.matrix.100tumorsamples) = top50muts

# 1. Perform hierarchical clustering on genomic alteration binary matrix
# Heatmap plot, information saved in variable: heatmap.plot
heatmap.plot = heatmap.2(mutation.matrix.100tumorsamples, trace='none', col=c(0,2))
# Extract row and column index
RowInd = heatmap.plot$rowInd
ColInd = heatmap.plot$colInd

# 2. Use the heatmap row and column index to reorganize the genomic alteration matrix, 
# which has the information of SM(1), CNA(+_2), SMandCNA(+_3).
mutation.matrix.100tumorsamples.hearmap.reorganize = mutation.matrix.100tumorsamples[RowInd, ]
mutation.matrix.100tumorsamples.hearmap.reorganize = mutation.matrix.100tumorsamples.hearmap.reorganize[, ColInd]


# 3. Replot the heatmap using 6 color codes to reflect different genomic alteration types.
image(t(mutation.matrix.100tumorsamples.hearmap.reorganize), col=c(0,2))


### BRCA top 100 SGAs clustering
# User defined function -- read csv data
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

install.packages("gplots")
library(gplots) 
# Read top 100 SGA binary matrix for BRCA 
BRCA.top100SGA.matrix = read.csvfile("BRCA/finalTop100Submatrix.csv")

# Read mutation info matrix for BRCA
BRCA.mutinfo.matrix = read.csvfile("/home/kevin/GroupDropbox/TCI/GtM.withmutinfo/BRCA.GtM.withmutinfo.csv")
# Reorganize the index to be the same as that of SGA binary matrix
BRCA.mutinfo.matrix = BRCA.mutinfo.matrix[,colnames(BRCA.top100SGA.matrix)]
BRCA.mutinfo.matrix = BRCA.mutinfo.matrix[rownames(BRCA.top100SGA.matrix),]

# Perform hierarchical clustering on genomic alteration binary matrix
heatmap.plot = heatmap.2(BRCA.top100SGA.matrix, trace='none', col=c(0,2), labRow=NULL, labCol=NULL, key=F, xlab=NULL, ylab=NULL, dendrogram="none")
RowInd = heatmap.plot$rowInd
ColInd = heatmap.plot$colInd

# Use the heatmap row and column index to reorganize the mutation information matrix, 
# which has the information of SM(1), CNA(+_2), SMandCNA(+_3).
mutation.matrix.100tumorsamples.hearmap.reorganize = mutation.matrix.100tumorsamples[RowInd, ]
mutation.matrix.100tumorsamples.hearmap.reorganize = mutation.matrix.100tumorsamples.hearmap.reorganize[, ColInd]









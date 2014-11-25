setwd("~/GroupDropbox/TCI/GtM.withmutinfo/")

barplotData = read.csv("BarplotNormalizedMatrix.csv")

barplotColNames = as.matrix(barplotData[-1, 1])
barplotRowNames = as.matrix(barplotData[1, -1])

barplotData.matrix = barplotData[-1,]
barplotData.matrix = barplotData.matrix[,-1]
barplotData.matrix = t(matrix(as.numeric(as.matrix(barplotData.matrix)), dim(barplotData.matrix)))
#write.csv(barplotData.matrix, file="~/GroupDropbox/TCI/DataMatricesForGraphs/barplotData.csv")

barplot(barplotData.matrix, xlab = "Tumor Type", ylab = "Normalized Frequency", col = 1:5, legend = barplotRowNames)


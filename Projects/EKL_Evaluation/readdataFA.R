library(EFAutilities)

source("spectral.R")

mnames = colnames(syms)

res2 = efa(x=syms, factors=2,  rotation="geomin", mnames=mnames)
plot(abs(res2$rotated))
res3 = efa(x=syms, factors=3,  rotation="geomin", mnames=mnames)
plot(abs(res3$rotated))

A <- cor(syms)
n <- nrow(A)
g <- colSums(A) # degrees of vertices
D_half = diag(1 / sqrt(g) )
L <- diag(n) - D_half %*% A %*% D_half 

K = 3
ei = eigen(A, symmetric = TRUE) 
ind = 1:K
#ind = order(ei$values[ei$values>0], decreasing = F)[1:K]
U <- ei$vectors[, ind]
rownames(U) <- mnames

df <- as.data.frame(U)

library(ggplot2)
# Simple scatter plot
ggplot(df, aes(V2, V3, label = rownames(df)))+
  geom_point()  + geom_text()
library(Rtsne)


tsne_out <- Rtsne(U,pca=TRUE,perplexity=8,theta=0)
tsne_value<-as.data.frame(tsne_out$Y)
p<-ggplot(tsne_value,aes(V1,V2, label = rownames(df)))+geom_point()+ geom_text()
p




K = 3
ei = eigen(L, symmetric = TRUE) 
#ind = 1:K
ind = order(ei$values[ei$values>0], decreasing = F)[1:K]
U <- ei$vectors[, ind]
rownames(U) <- mnames

df <- as.data.frame(U)

library(ggplot2)
# Simple scatter plot
ggplot(df, aes(V1, V2, label = rownames(df)))+
  geom_point()  + geom_text()


# Hierarchical clustering using Complete Linkage
d <- dist(U, method = "euclidean")
hc1 <- hclust(d, method = "complete")
# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)

library(factoextra)
clust <- cutree(hc1, k = 3)
fviz_cluster(list(data = U, cluster = clust))  ## from 'factoextra' package 


library(qgraph)
set.seed(0)
R = cor(syms)
K = 3
res2 <- SAAC(R, 3, m = 3)
combs = as.matrix(expand.grid(replicate(3, 0:1, simplify = FALSE)))
qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "C" = which(cls==5)), 
       color = c("lightblue", "lightsalmon", "lightcoral"))
cls <- apply(res2$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
Z = res2$Zones
ei = eigen(R, symmetric = TRUE) 
ind = 1:K
U <- ei$vectors[, ind]
X = solve(t(Z) %*% Z) %*% t(Z) %*% U
print(X)
B = X%*%t(X)
round(B, 2)
Z%*%B%*%t(Z)
# A = ZBZ'
# ZX = U
# X = Z^-1U
library("MPsychoR")
library("qgraph")
library("foreign")
library("GGMselect")

source("spectral.R")
set.seed(0)
# ----------------------- Rogers Depression & OCD data --------------------
##############################################
## contains symptoms for OCD and Depression ##
##    Labels are GIVEN (non-overlapping)    ##
## but we wanna see if symptoms can overlap ##
##############################################

load("Rogers.Rdata")
colnames(Rogers) <- 1:26

# plot data using correlation
adult_zeroorder <- cor(Rogers)
qgraph(adult_zeroorder, layout="spring", groups = list(Depression = 1:16, "OCD" = 17:26), 
       color = c("lightblue", "lightsalmon"))

# get adjacency matrix (GGM - from covariance matrix)
### To Do: method for ordinal variables ###
adj <- selectFast(as.matrix(Rogers), family = 'LA', K = 2.5)$LA$G

combs = as.matrix(expand.grid(replicate(2, 0:1, simplify = FALSE)))

set.seed(0)
res <- OCCAM(adj, 2, nonnegative = F)
cls <- apply(res$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(adult_zeroorder, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "no cluster" = which(cls==1)), 
       color = c("lightblue", "lightsalmon", "lightgray"))

res2 <- SAAC(adj, 2, m = 2)
cls <- apply(res2$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(adult_zeroorder, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3)), 
       color = c("lightblue", "lightsalmon"))

# ---------------------------- Schizophrenia data ------------------------
###################################################
##        symptoms for schizophrenia             ##
##        Positive vs. negative vs. general      ##
##        whether symptoms can overlap           ##
###################################################

schizo <-  read.csv('schizophrenia.csv')
col <- c(paste0("p",1:7), paste0("n",1:7), paste0("g",1:16))
syms <- schizo[,col]
syms[,"g8"] <- as.integer(syms[,"g8"])
syms <- syms[complete.cases(syms),] # keep complete cases only
print(dim(syms))

R <- cor(as.matrix(syms))
qgraph(R, layout="spring", groups = list("Positive" = 1:7, "Negative" = 8:14, "General" = 15:30), 
       color = c("lightblue", "lightsalmon", "lightcoral"))

penalty = 9
adj <- selectFast(as.matrix(syms), family = 'LA', K=penalty)$LA$G

# # two clusters
# combs = as.matrix(expand.grid(replicate(2, 0:1, simplify = FALSE)))
# 
# res <- OCCAM(adj, 2)
# cls <- apply(res$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
# qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3)), 
#        color = c("lightblue", "lightsalmon"))
# title(paste0("OCCAM (penalty =", penalty, ")"), line = 2.5)
# 
# res2 <- SAAC(adj, 2, m = 2)
# cls <- apply(res2$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
# qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3)), 
#        color = c("lightblue", "lightsalmon"))

# or three?
### TO DO: show the clusters for overlapping items, if possible ###
combs = as.matrix(expand.grid(replicate(3, 0:1, simplify = FALSE)))

set.seed(0)
res <- OCCAM(adj, 3)
cls <- apply(res$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "C" = which(cls==5)), 
       color = c("lightblue", "lightsalmon", "lightcoral"))

set.seed(0)
res2 <- SAAC(adj, 3, m = 3)
cls <- apply(res2$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "C" = which(cls==5)), 
       color = c("lightblue", "lightsalmon", "lightcoral"))


# ------------------------- Caring Emotions data -----------------------
#############################################################
##    State-Trait Scales for Distinct Positive Emotions    ##
##           STS-DPE was built by factor analysis          ##
##  Allowing overlaps in measuring psychological traits    ##
#############################################################

emo <- read.spss("Caring_Emotions.sav", use.value.labels = F, to.data.frame = T)

# Empathy, Sympathy, & Tenderness
col <- c(paste0('q0010_00',c('01','02',10,25,28,35,36)), paste0('q0012_00',c('06',16,17,18,33,43,44)),
            paste0('q0014_00',c('04',11,19,22,23,29,31,41)))
caring <- emo[,col]
caring <- caring[complete.cases(caring),]
col_name <- c(paste0('E',c(1,2,10,25,28,35,36)), paste0('S',c(06,16,17,18,33,43,44)),
              paste0('T',c('04',11,19,22,23,29,31,41)))
colnames(caring) <- col_name


R <- cor(caring)
qgraph(R, layout="spring", groups = list("Empathy" = 1:7, "Sympathy" = 8:14, "Tenderness" = 15:22), 
       color = c("lightblue", "lightsalmon", "lightcoral"))


adj <- selectFast(as.matrix(caring), family = 'LA', K=1.0)$LA$G

set.seed(0)
combs = as.matrix(expand.grid(replicate(3, 0:1, simplify = FALSE)))
res <- OCCAM(adj, 3, nonnegative = F)
cls <- apply(res$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "C" = which(cls==5), "no cluster" = which(cls==1)), 
       color = c("lightblue", "lightsalmon", "lightcoral","lightgray"))

set.seed(0)
res2 <- SAAC(adj, 3, m = 3)
cls <- apply(res2$Zones, 1, function(v) {which.min(rowSums(t(t(combs)-v)^2))})
qgraph(R, layout="spring", groups = list("A" = which(cls==2), "B" = which(cls==3), "C" = which(cls==5)), 
       color = c("lightblue", "lightsalmon", "lightcoral"))

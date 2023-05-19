#R version 3.6.3
library("igraph") #igraph version 1.2.5
library("sdpt3r") #sdpt3r version 0.3
library("doParallel") #doParallel version 1.0.15
library("matrixStats") #matrixStats version 0.56.0
library("MASS") #MASS version 7.3-51.5

setwd("Example 1") #set working directory

################################################################################
##1) Load data                                                                ##
################################################################################
#Note: Example 1 uses data from Michell and West "Peer pressure to smoke: the 
#meaning depends on the method." Can find data here:  
#<https://www.stats.ox.ac.uk/~snijders/siena/Glasgow_data.htm>.
#Example 1 uses data from years 1 and 3. 

load("Glasgow-friendship.RData")
load("Glasgow-selections.RData")

#Use the suggested sample of 129 students
sample = selection129

first = sign(friendship.1[sample,sample])
third = sign(friendship.3[sample,sample])

N = dim(first)[1] #number of agents 


################################################################################
##2) Summary statistics                                                       ##
################################################################################
#first convert adjacency matrices to igraph objects
graphFirst = graph.adjacency(first)
graphThird = graph.adjacency(third)
#degree
mean(rowSums(first))
sd(rowSums(first))
mean(rowSums(third)) 
sd(rowSums(third))
#eigenvector centrality
mean(as.vector(eigen_centrality(graphFirst)$vector))
sd(as.vector(eigen_centrality(graphFirst)$vector))
mean(as.vector(eigen_centrality(graphThird)$vector))
sd(as.vector(eigen_centrality(graphThird)$vector))
#clustering
transitivity(graphFirst, type = "global")
transitivity(graphThird, type = "global")
#diameter
diameter(graphFirst)
diameter(graphThird)

################################################################################
##3) Regressions                                                              ##
################################################################################

X = cbind(as.vector(matrix(1,2*N,1)),
           c(as.vector(matrix(0,N,1)),as.vector(matrix(1,N,1))))
Y1 = c(rowSums(first),rowSums(third))
Y2 = c(as.vector(eigen_centrality(graphFirst)$vector),
       as.vector(eigen_centrality(graphThird)$vector))
Y3 = c(transitivity(graphFirst, type = "local"),
       transitivity(graphThird, type = "local"))

beta1 = ginv(t(X)%*%X)%*%(t(X)%*%Y1)
variance1 = as.double(t(Y1-X%*%beta1)%*%(Y1-X%*%beta1))*ginv(t(X)%*%X)/(2*N-2)

beta2 = ginv(t(X)%*%X)%*%(t(X)%*%Y2)
variance2 = as.double(t(Y2-X%*%beta2)%*%(Y2-X%*%beta2))*ginv(t(X)%*%X)/(2*N-2)

X3 = X[is.nan(Y3) == 0,]
Y3 = Y3[is.nan(Y3) == 0]
beta3 = ginv(t(X3)%*%X3)%*%(t(X3)%*%Y3)
variance3 = as.double(t(Y3-X3%*%beta3)%*%(Y3-X3%*%beta3))*
  ginv(t(X3)%*%X3)/(2*N-2)

tstat11 = beta1[1]^2/(variance1[1,1])
c(beta1[1],1-pchisq(tstat11,1))
tstat12 = beta1[2]^2/(variance1[2,2])
c(beta1[2],1-pchisq(tstat12,1))

tstat21 = beta2[1]^2/(variance2[1,1])
c(beta2[1],1-pchisq(tstat21,1))
tstat22 = beta2[2]^2/(variance2[2,2])
c(beta2[2],1-pchisq(tstat22,1))

tstat31 = beta3[1]^2/(variance3[1,1])
c(beta3[1],1-pchisq(tstat31,1))
tstat32 = beta3[2]^2/(variance3[2,2])
c(beta3[2],1-pchisq(tstat32,1))

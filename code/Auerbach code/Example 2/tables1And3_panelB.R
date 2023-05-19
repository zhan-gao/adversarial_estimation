#R version 3.6.3
library("igraph") #igraph version 1.2.5
library("sdpt3r") #sdpt3r version 0.3
library("doParallel") #doParallel version 1.0.15
library("matrixStats") #matrixStats version 0.56.0
library("MASS") #MASS version 7.3-51.5

setwd("Example 2") #set working directory

################################################################################
##1) Load data                                                                ##
################################################################################
#Example 2 uses data from Banerjee, Chandrasekhar, Duflo, and Jackson (2013): 
#"Diffusion of Microfinance." Can find data here: 
#https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U3BIHX.
#Example 2 uses data from village 10.
file.list = list.files(, pattern = ".csv", full.names = T) #list networks
allmatrix = list()
x=1
for (i in 1:(length(file.list))){
  allmatrix[[x]] = as.matrix((read.csv(file.list[[i]],
                                       header = FALSE, sep = ",")))
  x=x+1
}
#Example compares two social and economic networks between villagers
#social network are self-identified friendships
social = allmatrix[[8]] 
#economic network are self-identified economic links: borrowing or lending 
#money, rice, or kersosene
econ = pmax(allmatrix[[1]],allmatrix[[6]],allmatrix[[4]],allmatrix[[5]]) 
#N is the number of agents  
N = dim(social)[1] #number of agents 


################################################################################
##2) Summary statistics                                                       ##
################################################################################
#first convert adjacency matrices to igraph objects
graphSocial = graph.adjacency(social)
graphEcon = graph.adjacency(econ)
#degree
mean(rowSums(social))
sd(rowSums(social))
mean(rowSums(econ)) 
sd(rowSums(econ))
#eigenvector centrality
mean(as.vector(eigen_centrality(graphSocial)$vector))
sd(as.vector(eigen_centrality(graphSocial)$vector))
mean(as.vector(eigen_centrality(graphEcon)$vector))
sd(as.vector(eigen_centrality(graphEcon)$vector))
#clustering
transitivity(graphSocial, type = "global")
transitivity(graphEcon, type = "global")
#diameter
diameter(graphSocial)
diameter(graphEcon)

################################################################################
##3) Regressions                                                              ##
################################################################################

X = cbind(as.vector(matrix(1,2*N,1)),
          c(as.vector(matrix(0,N,1)),as.vector(matrix(1,N,1))))
Y1 = c(rowSums(social),rowSums(econ))
Y2 = c(as.vector(eigen_centrality(graphSocial)$vector),
       as.vector(eigen_centrality(graphEcon)$vector))
Y3 = c(transitivity(graphSocial, type = "local"),
       transitivity(graphEcon, type = "local"))

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

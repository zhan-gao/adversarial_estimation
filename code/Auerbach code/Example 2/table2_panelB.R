#R version 3.6.3
library("igraph") #igraph version 1.2.5
library("sdpt3r") #sdpt3r version 0.3
library("doParallel") #doParallel version 1.0.15
library("matrixStats") #matrixStats version 0.56.0
start = Sys.time()
################################################################################
##This code recreates Example 2 from Auerbach (2020):                         ##
##"Testing for Differences in Stochastic Network Structure"                   ##
##https://arxiv.org/pdf/1903.11117.pdf                                        ##                   
################################################################################
#To run the code do these three things. 
setwd("Example 2") #set working directory
R = 499  #choose the number of simulations for the randomization tests
numCores = 7 #choose number of cores for paralellization    
registerDoParallel(numCores)

#Approximate runtime is 25*(R+1)/numCores seconds.

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
totDeg = abs(mean(social)-mean(econ)) 
diffDeg = mean((rowSums(social)-rowSums(econ))^2)   
#eigenvector centrality
centSocial = as.vector(eigen_centrality(graphSocial)$vector)
centEcon = as.vector(eigen_centrality(graphEcon)$vector)
diffCent = mean((as.vector(eigen_centrality(graph.adjacency(social))$vector)
                 -as.vector(eigen_centrality(graph.adjacency(econ))$vector))^2)                      
#clustering
diffTran = abs(transitivity(graphEcon, type = "global") - 
                 transitivity(graphSocial, type = "global"))
#diameter
diffDiam = abs(diameter(graphEcon)-diameter(graphSocial))
#all the statistics together
stats = c(totDeg,diffDeg,diffCent,diffTran,diffDiam) 
################################################################################
##3) Randomization tests using summary  statistics                            ##
################################################################################
#simulations
permuteStats <- foreach(r = 1:R, .combine='rbind', 
        .packages = c("igraph","foreach","MASS","Matrix", "sdpt3r")) %dopar%  {  
    v = matrix(sign(runif(N^2,-1,1)),N)  #matrix of rademachers
    v = upper.tri(v, diag = FALSE)*v +
                  t(upper.tri(v,diag = FALSE)*v) #make symmetric
                          
permuteSocial = pmax(social*(v == 1),econ*(v==-1))
permuteEcon = pmax(econ*(v == 1),social*(v==-1))
permuteTotDeg = abs(mean(permuteSocial)-mean(permuteEcon))
permuteDiffDeg = mean((rowSums(permuteSocial)-rowSums(permuteEcon))^2)                    
permuteDiffCent = mean((as.vector(eigen_centrality(
  graph.adjacency(permuteSocial))$vector)-
  as.vector(eigen_centrality(graph.adjacency(permuteEcon))$vector))^2)  
pemuteDiffTran = abs(transitivity(graph.adjacency(permuteEcon), type = "global") 
                - transitivity(graph.adjacency(permuteSocial), type = "global"))
permuteDiffDiam = abs(diameter(graph.adjacency(permuteEcon))-
   diameter(graph.adjacency(permuteSocial)))
                          
c(permuteTotDeg,permuteDiffDeg,permuteDiffCent,pemuteDiffTran,permuteDiffDiam)
        }
permuteStats = rbind(permuteStats,stats) 

#results
stats
t(colQuantiles(permuteStats,probs = c(.5,.75,.9,.95,.97,.99)))
c(mean(permuteStats[,1] >= stats[1]),mean(permuteStats[,2] >= stats[2]),
  mean(permuteStats[,3] >= stats[3]),mean(permuteStats[,4] >= stats[4]),
  mean(permuteStats[,5] >= stats[5]))

################################################################################
##4) Randomization test using the 2-2 norm                                    ##
################################################################################
#2-2 norm
spectralNorm = max(svd((social-econ))$d) 
#simulations
permuteSpectral = foreach(r = 1:R, .combine='rbind', 
        .packages = c("igraph","foreach","MASS","Matrix", "sdpt3r")) %dopar%  {  
rm(.Random.seed, envir=globalenv())                            
v = matrix(sign(runif(N^2,-1,1)),N)  #matrix of rademachers
v = upper.tri(v, diag = FALSE)*v + 
      t(upper.tri(v,diag = FALSE)*v) #make symmetric
                          
permuteSocial = pmax(social*(v == 1),econ*(v==-1))
permuteEcon = pmax(econ*(v == 1),social*(v==-1))
permuteSpectralNorm = max(svd((permuteSocial-permuteEcon))$d)
        }
permuteSpectral = rbind(permuteSpectral,spectralNorm) 
#results
spectralNorm #2-2 norm evaluated on data
cbind(colQuantiles(permuteSpectral,probs 
    = c(.5,.75,.9,.95,.97,.99))) #quantiles of reference distribution
mean(spectralNorm <= permuteSpectral) #pvalue
#histogram of reference distribution. Vertical line at data. 
hist(permuteSpectral/(sqrt(max(rowSums((social-econ)^2)))), breaks = 100, main = 
       "Histogram for 2-2 norm", xlab = NULL,ylab = NULL) 
abline(v = spectralNorm/(sqrt(max(rowSums((social-econ)^2)))), col = "red") 

################################################################################
##5) Randomization test using the infty-1 norm                                ##
################################################################################
#SD approx to infty-1 norm: see paper for derivation and sqlp function in 
#SDPT3R package for syntax. SDPT3R comes from 
#Toh, Todd, and Tutuncu (2003):"Solving semidefinite-quadratic-linear programs 
#using SDPT3."
bigDiffMatrix = rbind(cbind((social-econ)*0,(social-econ)),
    cbind(t((social-econ)),(social-econ)*0))/2 
blk = c("s" = (2*N))
one = matrix(1,nrow= 2*N,ncol=1)
A = matrix(list(),nrow=1,ncol=(2*N))  
for(k in 1:(2*N)){
  A[[k]] <- matrix(0,nrow=(2*N),ncol=(2*N))
  diag(A[[k]])[k] = 1
}
At = svec(blk[1],A,1)
b = matrix(1,nrow=2*N,ncol=1)
#SD approx to infty-1 norm
cutNorm = abs(sqlp(blk, At, list(bigDiffMatrix), b)$pobj) 
#simulations
permuteCut = foreach(r = 1:R, .combine='rbind', 
        .packages = c("igraph","foreach","MASS","Matrix", "sdpt3r")) %dopar%  {  
rm(.Random.seed, envir=globalenv())                                                   
v = matrix(sign(runif(N^2,-1,1)),N)  #matrix of rademachers
v = upper.tri(v, diag = FALSE)*v +
    t(upper.tri(v,diag = FALSE)*v) #make symmetric
permuteSocial = pmax(social*(v == 1),econ*(v==-1))
permuteEcon = pmax(econ*(v == 1),social*(v==-1))
                            
permuteBigDiffMatrix = rbind(cbind((permuteSocial-permuteEcon)*0,
    (permuteSocial-permuteEcon)),cbind(t((permuteSocial-permuteEcon)),
    (permuteSocial-permuteEcon)*0))/2
permuteCutNorm = abs(sqlp(blk, At, list(permuteBigDiffMatrix), b)$pobj)
        }
permuteCut = rbind(permuteCut,cutNorm) 
#results
cutNorm #infty-1 norm evaluated on data
cbind(colQuantiles(permuteCut,probs 
    = c(.5,.75,.9,.95,.97,.99))) #reference distribution
mean(cutNorm <= permuteCut) #pvalue
#histogram of the reference distribution. Veritcal line at data
hist(permuteCut/(sum(sqrt(rowSums((social-econ)^2)))),breaks = 100, main = 
       "Histogram for infty-1 norm", xlab = NULL, ylab = NULL)
abline(v = cutNorm/(sum(sqrt(rowSums((social-econ)^2)))), col = "red")
end = Sys.time()
end-start #runtime

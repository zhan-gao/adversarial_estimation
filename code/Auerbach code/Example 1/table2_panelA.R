#R version 3.6.3
library("igraph") #igraph version 1.2.5
library("sdpt3r") #sdpt3r version 0.3
library("doParallel") #doParallel version 1.0.15
library("matrixStats") #matrixStats version 0.56.0
start = Sys.time()
################################################################################
##This code recreates Example 1 from Auerbach (2020):                         ##
##"Testing for Differences in Stochastic Network Structure"                   ##
##https://arxiv.org/pdf/1903.11117.pdf                                        ##                   
################################################################################
#To run the code do these three things. 
setwd("Example 1") #set working directory
R = 499  #choose the number of simulations for the randomization tests
numCores = 7 #choose number of cores for paralellization    
registerDoParallel(numCores)

#Approximate runtime is 140*(R+1)/numCores seconds.

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
totDeg = abs(mean(first)-mean(third)) 
diffDeg = mean((rowSums(first)-rowSums(third))^2)   
#eigenvector centrality
centFirst = as.vector(eigen_centrality(graphFirst)$vector)
centThird = as.vector(eigen_centrality(graphThird)$vector)
diffCent = mean((as.vector(eigen_centrality(graph.adjacency(first))$vector)
                 -as.vector(eigen_centrality(graph.adjacency(third))$vector))^2)                      
#clustering
diffTran = abs(transitivity(graphThird, type = "global") - 
                 transitivity(graphFirst, type = "global"))
#diameter
diffDiam = abs(diameter(graphThird)-diameter(graphFirst))
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
                          
permuteFirst = pmax(first*(v == 1),third*(v==-1))
permuteThird = pmax(third*(v == 1),first*(v==-1))
permuteTotDeg = abs(mean(permuteFirst)-mean(permuteThird))
permuteDiffDeg = mean((rowSums(permuteFirst)-rowSums(permuteThird))^2)                    
permuteDiffCent = mean((as.vector(eigen_centrality(
   graph.adjacency(permuteFirst))$vector)-
   as.vector(eigen_centrality(graph.adjacency(permuteThird))$vector))^2)  
pemuteDiffTran = abs(transitivity(graph.adjacency(permuteThird),type = "global") 
  - transitivity(graph.adjacency(permuteFirst), type = "global"))
permuteDiffDiam = abs(diameter(graph.adjacency(permuteThird))-
   diameter(graph.adjacency(permuteFirst)))
                          
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
spectralNorm = max(svd((first-third))$d) 
#simulations
permuteSpectral = foreach(r = 1:R, .combine='rbind', 
     .packages = c("igraph","foreach","MASS","Matrix", "sdpt3r")) %dopar%  {  
      rm(.Random.seed, envir=globalenv())                            
   v = matrix(sign(runif(N^2,-1,1)),N)  #matrix of rademachers
   v = upper.tri(v, diag = FALSE)*v + 
                 t(upper.tri(v,diag = FALSE)*v) #make symmetric
                            
permuteFirst = pmax(first*(v == 1),third*(v==-1))
permuteThird = pmax(third*(v == 1),first*(v==-1))
permuteSpectralNorm = max(svd((permuteFirst-permuteThird))$d)
                          }
permuteSpectral = rbind(permuteSpectral,spectralNorm) 
#results
spectralNorm #2-2 norm evaluated on data
cbind(colQuantiles(permuteSpectral,probs 
                   = c(.5,.75,.9,.95,.97,.99))) #quantiles of reference distribution
mean(spectralNorm <= permuteSpectral) #pvalue
#histogram of reference distribution. Vertical line at data. 
hist(permuteSpectral/(sqrt(max(rowSums((first-third)^2)))), breaks = 100, main = 
       "Histogram for 2-2 norm", xlab = NULL,ylab = NULL) 
abline(v = spectralNorm/(sqrt(max(rowSums((first-third)^2)))), col = "red") 

################################################################################
##5) Randomization test using the infty-1 norm                                ##
################################################################################
#SD approx to infty-1 norm: see paper for derivation and sqlp function in 
#SDPT3R package for syntax. SDPT3R comes from 
#Toh, Todd, and Tutuncu (2003):"Solving semidefinite-quadratic-linear programs 
#using SDPT3."
bigDiffMatrix = rbind(cbind((first-third)*0,(first-third)),
                      cbind(t((first-third)),(first-third)*0))/2 
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
permuteFirst = pmax(first*(v == 1),third*(v==-1))
permuteThird = pmax(third*(v == 1),first*(v==-1))
                       
permuteBigDiffMatrix = rbind(cbind((permuteFirst-permuteThird)*0,
       (permuteFirst-permuteThird)),cbind(t((permuteFirst-permuteThird)),
       (permuteFirst-permuteThird)*0))/2
permuteCutNorm = abs(sqlp(blk, At, list(permuteBigDiffMatrix), b)$pobj)
                     }
permuteCut = rbind(permuteCut,cutNorm) 
#results
cutNorm #infty-1 norm evaluated on data
cbind(colQuantiles(permuteCut,probs 
                   = c(.5,.75,.9,.95,.97,.99))) #reference distribution
mean(cutNorm <= permuteCut) #pvalue
#histogram of the reference distribution. Veritcal line at data
hist(permuteCut/(sum(sqrt(rowSums((first-third)^2)))),breaks = 100, main = 
       "Histogram for infty-1 norm", xlab = NULL, ylab = NULL)
abline(v = cutNorm/(sum(sqrt(rowSums((first-third)^2)))), col = "red")
end = Sys.time()
end-start #runtime

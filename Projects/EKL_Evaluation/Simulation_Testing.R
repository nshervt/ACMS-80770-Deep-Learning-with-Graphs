library('gtools')
library('perturbR')


###--------- Calculate Estimation Error ---------

# X1 is the estimated matrix, and X2 is the true matrix
Estim_error <- function(X1, X2){
  n <- nrow(X1)
  K <- ncol(X2)
  
  perm <- permutations(K, K)
  dist <- rep(0, nrow(perm))
  for (i in 1:nrow(perm)){
    P <- matrix(rep(0,K*K),K,K)
    
    for (j in 1:nrow(P)){
      P[j,perm[i,j]] <- 1
      }
    
    mat <- X2 %*% P - X1
    dist[i] <- sum(mat^2)
  }
  error <- min(dist)/(n*K)
  return(error)
}

###--------- Calculate Normalized Variation of Information (NVI) ----------
entropy <- function(x, k){
  P0 <- sum(x[,k] == 0)/nrow(x)
  P1 <- sum(x[,k] == 1)/nrow(x)
  
  H <- -P0*log(P0) - P1*log(P1)
  return(H)
}

joint_entropy <- function(x1, x2, k, sigma_k){
  
  P00 <- sum(x1[,k] == 0 & x2[,sigma_k] == 0)/nrow(x1)
  P01 <- sum(x1[,k] == 0 & x2[,sigma_k] == 1)/nrow(x1)
  P10 <- sum(x1[,k] == 1 & x2[,sigma_k] == 0)/nrow(x1)
  P11 <- sum(x1[,k] == 1 & x2[,sigma_k] == 1)/nrow(x1)
  
  if (P00 != 0) item1 <- - P00*log(P00) else item1 <- 0
  if (P01 != 0) item2 <- - P01*log(P01) else item2 <- 0
  if (P10 != 0) item3 <- - P10*log(P10) else item3 <- 0
  if (P11 != 0) item4 <- - P11*log(P11) else item4 <- 0
  
  H <- item1 + item2 + item3 + item4
  return(H)
}

NVI <- function(x1, x2){
  n <- nrow(x1)
  K <- ncol(x2)
  
  perm <- permutations(K, K)
  dist <- rep(0, nrow(perm))
  for (i in 1:nrow(perm)){
    sum <- 0
    for (j in 1:K){
      
      H.XY <- joint_entropy(x1, x2, j, perm[i,j])
      H.Y <- entropy(x2,perm[i,j])
      H.X <- entropy(x1,j)
      
      sum <- sum + (H.XY - H.Y)/H.X + (H.XY - H.X)/H.Y
    }
    dist[i] <- sum/(2*K)
  }
  NVI <- 1 - min(dist)
  return(NVI)
}


library(Gmedian)
library(ClusterR)

OCCAM <- function(A, K=2, nonnegative=F ,cls=NA) {
  n <- nrow(A)
  alpha <- (sum(A) - sum(diag(A))) / n / (n-1) / K
  tau <- 0.1 * (alpha^0.2) * (K^1.5) / (n^0.3)
  #g <- colSums(A) # degrees of vertices
  #D_half = diag(1 / sqrt(g) )
  #A2 <- diag(n) - D_half %*% A %*% D_half 
  #L <- diag(g) - W
  
  
  #ei = eigen(A2, symmetric = TRUE) 
  ei = eigen(A, symmetric = TRUE) 
  ind = 1:K
  #ind = order(ei$values[ei$values>0], decreasing = T)[1:K]
  #U <- ei$vectors[,(n - K):(n - 1)]
  U <- ei$vectors[, ind]
  #L <- diag(ei$values[(n - K):(n - 1)])
  L <- diag(ei$values[ind])
  X_hat <- U %*% sqrt(L)
  X_hat <- t(apply(X_hat, 1, function(x) x/(norm(x, type="2") + tau)))
  
  cl.kmedian <- kGmedian(X_hat, K)
  S <- cl.kmedian$centers
  
  X_hat[X_hat==0] = 0.0000001
  
  
  #z S - X_hat = 0
  # (z S - X_hati) (z S - X_hai)'
  # ( X_hati - zS)(X_hai - zS)'
  # ( X_hati' - S'z')'(X_hai' - S'z')
  # grad = -2 S (X_hati' - S'z')
  
  
  invS = solve(S)
  Z = X_hat %*% invS
  if (nonnegative) {
    for (i in 1:n) {
      if (sum(Z[i,]<0)>0) {
        er = function (z){
          z = matrix(z, nrow=1, ncol=K)
          x = matrix(X_hat[i,], nrow=1, ncol=K)
          v = (z%*%S - x)
          return(v %*% t(v))
        }
        
        der = function (z){
          z = matrix(z, nrow=1, ncol=K)
          x = matrix(X_hat[i,], nrow=1, ncol=K)
          return(-2 * S %*% (t(x) - t(S)%*%t(z)))
        }
        Z[i,] = constrOptim(rep(0.2,K), er, grad=der, ui=diag(K), ci=rep(0,K))$par
      }
    }
  }
  Z = t(apply(Z, 1, function(x) x/(norm(x, type="2"))))
  
  if (is.na(cls)) {
    par(mfrow = c(1, 2))
  } else {
    par(mfrow = c(1, 3))
    plot(X_hat, col=cls)
    title("True")
  }
  
  plot(X_hat, col=apply(Z,1,which.max))
  title("Estimtaed Proj")
  plot(X_hat, col=cl.kmedian$cluster)
  title("Estimtaed KM")
  #plot(X_hat, col=cls)
  #plot(X, col=cl.kmedian$cluster)
  Zones <- Z
  Zones[Zones>1/K] <- 1
  Zones[Zones<=1/K] <- 0
  cls <- apply(Z,1,which.max)
  
  # cls should only be used when there is no overlapping communities.
  list(Z=Z, Zones=Zones, cluster=cls, X_hat=X_hat)
}

# m is the maximum overlap
SAAC  <- function(A, K=-1, m = 3, eta = 0.1, r = 0.3, cls=NA) {
  n <- nrow(A)
  dmax <- max(rowSums(A))
  ei <- eigen(A, symmetric = TRUE) 
  
  # Adaptive if K=-1 is supplied
  if (K < 1) {
    lambdamin <- sqrt(2*(1+eta)*dmax*log(4*n^(1+r)))
    K <- sum(ei$values > lambdamin)
  }
  
  ind <- order(abs(ei$values), decreasing = T)[1:K]
  U <- ei$vectors[, ind]
  Z <- matrix(0, nrow=n, ncol=K)
  km <-  KMeans_rcpp(U, clusters = K, initializer = 'kmeans++')
  X <- km$centroids
  comb <- as.matrix(expand.grid(replicate(K, 0:1, simplify = FALSE)))
  comb <- comb[1 <= rowSums(comb) & rowSums(comb) <= m,]
  temp <- rep(0, nrow(comb))
  loss <- Inf
  nloss <- sum((U - Z %*% X)^2)
  while (loss - nloss > 0.0001) {
    loss <- nloss
    for (i in 1:n) {
      for (j in 1:nrow(comb)) {
        temp[j] <- sum((U[i,] - comb[j,] %*% X)^2)
      }
      Z[i,] <- comb[which.min(temp),]
    }
    X = solve(t(Z) %*% Z) %*% t(Z) %*% U
    nloss <- sum((U - Z %*% X)^2)
  }
  
  if (is.na(cls)) {
    par(mfrow = c(1, 1))
  } else {
    par(mfrow = c(1, 2))
    plot(U, col=cls)
    title("True")
  }
  
  plot(U, col=apply(Z,1,which.max))
  title("Estimated")
  print(nloss)
  cls <- apply(Z,1,which.max)
  # cls should only be used when there is no overlapping communities.
  list(Z=Z, Zones=Z, cluster=cls, X=X, B=X%*%t(X))
}





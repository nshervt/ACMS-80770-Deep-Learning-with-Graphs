
##############
##############
#     SBM    #
##############
##############


sample_adj = function(A_hat) {
  n = nrow(A_hat)
  A = matrix(0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      p = 0
      if (A_hat[i,j] > 1) {
        p = 1
      } else {
        p = rbinom(1, 1, A_hat[i,j])
      }
      A[i,j] = p
      A[j,i] = p
    }
  }
  A
}

gen_SBM_testdata = function(n){
  K = 3
  B = matrix(c(0.7, 0.3, 0.1, 0.3, 0.7, 0.2, 0.1, 0.2, 0.7), 3, 3)
  comb = as.matrix(expand.grid(replicate(K, 0:1, simplify = FALSE)))[2:8,]
  probs = c(0.4, 0.2, 0.05, 0.2, 0.05, 0.09, 0.01)
  
  category = sample(1:length(probs), n, T)
  
  Z = matrix(0, nrow = n, ncol = K)
  
  for (i in 1:n) {
    Z[i,] <- comb[category[i], ]
  }
  
  A_e  = (Z %*% B %*% t(Z))
  alpha = (sum(A_e) - sum(diag(A_e)))/n/(n-1)/K
  A_e = A_e*alpha
  
  
  A = sample_adj(A_e)
  
  #A[A > 0.1] = 1
  #A[A<=0.1]=0
  
  list(A_e=A_e, A=A, Z=Z, B=B)
}



#############################
#############################
#    latent space models    #
#############################
#############################

gen_LSM_testdata = function(n){
  
  z1 = cbind(runif(n/2, 0, 2), runif(n/2, 0, 2))
  z2 = cbind(runif(n/2, 1, 3), runif(n/2, 1, 3))
  z = rbind(z1,z2)
  Z = matrix(0, ncol=2, nrow=n)
  for (i in 1:n) {
    if (i <= n/2) {
      Z[i,1] = 1
    } else {
      Z[i,2] = 1
    }
    if (1 <= z[i,1] && 1 <= z[i,2] && z[i,1] <= 2 && z[i,2] <= 2) {
      Z[i,1] = Z[i,2] = 1
    }
  }
  plot(rbind(z), col=rowSums(Z)) # visualize the overlapping part
  
  A_e = matrix(0, n, n)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      A_e[i,j] = exp(norm(z[i,] - z[j,], type="2")) / (1 + exp(norm(z[i,] - z[j,], type="2")))
    }
  }
  A = sample_adj(A_e)
  
  list(A_e=A_e, A=A, Z=Z, z=z)
}

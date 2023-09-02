# cd /pine/scr/y/s/yshan/p3/data/sim/dat

tr = function(A){
  sum(diag(A))
}

x = list()
sumRk_inv = matrix(0, nrow=3, ncol=3)
s1 = s2 = matrix(0, nrow = 3, ncol = 3)
g1 = diag(c(1,0))
g2 = diag(c(0,1))
for(k in 1:4){
  x[[k]] = as.matrix(read.table(paste0('x/x',k,'.txt')))
  sumRk_inv = sumRk_inv + solve(t(x[[k]])%*%x[[k]])
  tmp = t(x[[k]]) %*% x[[k]][,-1]
  s1 = s1 + tmp %*% g1 %*% t(tmp)
  s2 = s2 + tmp %*% g2 %*% t(tmp)
}
x0 = as.matrix(read.table('x/x0.txt'))

x.all = rbind(x[[1]],x[[2]],x[[3]],x[[4]])
R_inv = solve(t(x.all)%*%x.all)
R0 = t(x0) %*% x0
d0 = tr((sumRk_inv/16 - R_inv) %*% R0) 
z0z0 = t(x0[,-1])%*%x0[,-1]

A = read.table('A.txt')[2,2]

d2 = tr(R_inv %*% s1 %*% R_inv %*% R0) - z0z0[1,1]/4
t2 = d0/d2
thr2 = t2 * A # 0.5916722 [sigma_1^2 + sigma_2^2]

d3 = tr(R_inv %*% s2 %*% R_inv %*% R0) - z0z0[2,2]/4
t3 = d0/d3
thr3 = t3 * A # 0.4153981 [sigma_1^2 + sigma_2^2]

h = 0.07
o = cbind(h,A,thr2,thr3)
write.table(o,'thr_sig2_uneq.txt', row.names = F, sep = '\t', quote = F)

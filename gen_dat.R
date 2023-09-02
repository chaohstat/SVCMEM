# cd /pine/scr/y/s/yshan/p3/data/sim/dat
# generate data for Uneq h3
# y_v0/k{0..4}.r: nrep=1000

hi = 3
p.y = paste0('y_uneq_h',hi,'/')
if(!file.exists(p.y)){
  system(paste0('mkdir ', p.y))
}

# thr = unlist(read.table('thr_sig2_uneq.txt', header = T)[3:4])
# thr.itv = thr[2]/3
# sig2 = c((1:3)*thr.itv, mean(thr), thr[1], (1:4)*thr.itv + thr[1]) # sigma^2
# write.table(sig2, paste0('sig2_uneq_h',hi,'.txt'), row.names = F, col.names = F)

sig2 = scan(paste0('sig2_uneq_h',hi,'.txt'))

x = list()
for(k in 1:4){
  x[[k]] = as.matrix(read.table(paste0('x/x',k,'.txt')))
}
x0 = as.matrix(read.table('x/x0.txt'))
x[[5]] = x0[1:100,]
x[[6]] = x0[101:200,]
x[[7]] = x0[201:300,]
x[[8]] = x0[301:400,]
b = as.matrix(read.table('beta.txt'))
C = as.matrix(read.table('C.txt'))
psi = as.matrix(read.table('psi.txt'))


lambda1 = 0.4 # use this same lambda v.2
lambda2 = 0.2
n = 100
M = 50
n_rep = 5000

for(s in 1:9){
  message(s)
  for(r in 1:n_rep){
    if(r%%200==0){ cat(r,'') }
    y0 = NULL
    for(k in 1:8){
      xi1 = rnorm(n,0,sqrt(lambda1))
      xi2 = rnorm(n,0,sqrt(lambda2))
      eta = xi1%*%t(psi[,1]) + xi2%*%t(psi[,2]) # n x M, for one replicate, one study (out of 5 = 4 train + 1 test) [eta same for eq.v & uneq.v]
      epsilon = matrix(rnorm(n*M, 0, sqrt(0.2)), ncol = M)   # n x M
      c_gamma = c(rnorm(1, 0, sqrt(sig2[s]/3)),
                  rnorm(1, 0, sqrt(2*sig2[s]/3)))
      gamma = matrix(rep(c(0,c_gamma),50), nrow=50, byrow = T)               # 50 x 3: repeat c_gamma across v
      beta = b %*% diag(C[r,]) + gamma                                       # 50 x 3
      y = rbind(x[[k]]) %*% t(beta) + eta + epsilon     # 100 x 50
      if(k < 5){
        write.table(y[1:n,], paste0(p.y,'v',s,'_k',k,'.',r), row.names = F, col.names = F, sep = '\t')
      } else {
        y0 = rbind(y0, y)
      }
    }
    write.table(y0, paste0(p.y, 'v',s,'_k0.',r), row.names = F, col.names = F, sep = '\t')
  }
  cat('\n')
}

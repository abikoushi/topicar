make_poem <- function(U,W,B,vocab,seed=NULL){
  if(is.null(seed)){
    seed <- sample.int(.Machine$integer.max,1)
  }
  set.seed(seed)
  n_vocab <- length(vocab)
  N <- nrow(U)
  W <- t(W)
  s <- integer(N)
  z <- t(rmultinom(1,1,exp(U[1,]%*%W)))
  s[1] <- which(z==1)
  for (i in 2:N) {
    z <- t(rmultinom(1,1,exp(U[i,]%*%W+z%*%B)))
    s[i] <- which(z==1)
  }
  list(poem=paste(vocab[s],collapse = ""),
       seed=seed)
}

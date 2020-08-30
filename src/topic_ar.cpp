// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <progress.hpp>
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;

//activation//
class activation {
public:
  virtual arma::mat eval(arma::mat x) = 0; 
  virtual ~activation(){}
};

class softmax : public activation{
public:
  arma::mat eval(arma::mat x){
    arma::mat ex = exp(x.each_row() - max(x,0));
    return ex.each_row() / sum(ex,0);
  }
};

class logistic : public activation{
public:
  arma::mat eval(arma::mat x){
    return 1.0/(1.0+exp(-x));
  }
};

class softplus : public activation{
public:
  arma::mat eval(arma::mat x){
    return log1p(exp(x));
  }
};

class sigmoid : public activation{
public:
  arma::mat eval(arma::mat x){
    return tanh(x);
  }
};

class identity : public activation{
public:
  arma::mat eval(arma::mat x){
    return x;
  }
};

arma::mat col_softmax(const arma::mat & x){
  arma::mat ex = exp(x.each_row() - max(x,0));
  return ex.each_row() / sum(ex,0);
}

arma::mat logcol_softmax(const arma::mat & x){
  return x.each_row() - log(sum(exp(x),0));
}

arma::uvec sample_arma(arma::vec & pvec) {
  arma::uword K = pvec.n_elem;
  arma::uvec opts = arma::linspace<arma::uvec>(0L, K - 1L, K);
  return Rcpp::RcppArmadillo::sample(opts, K, true, pvec);
}

// [[Rcpp::export]]
List topic_ar(arma::sp_mat X, double sigma, int L, int n_p, int iter, double lr, std::string actfun){
  int N = X.n_rows;
  int K = X.n_cols;
  // X.insert_rows(0,arma::zeros<arma::rowvec>(K));
  std::unique_ptr<activation> g;
  if(actfun=="softmax"){
    g.reset(new softmax);
  }else if(actfun=="softplus"){
    g.reset(new softplus);
  }else if(actfun=="logistic"){
    g.reset(new logistic);
  }else if(actfun=="sigmoid"){
    g.reset(new sigmoid);
  }else if(actfun=="identity"){
    g.reset(new identity);
  }else{
    stop("this activation function is not implemented\n");
  }
  arma::mat W = arma::randn<arma::mat>(K,L);
  arma::mat B = arma::zeros<arma::mat>(K,K);
  arma::cube Z(N,L,n_p);
  arma::cube U(N,L,n_p);
  arma::vec loglik = arma::zeros<arma::vec>(iter);
  Progress prog(iter);
  for(int i=0; i<iter; i++){
    Z.randn();
    Z = Z*sigma;
    arma::mat d_W = arma::zeros<arma::mat>(K,L);
    arma::mat d_B = arma::zeros<arma::mat>(K,K);
    for(int n=1; n<N; n++){
      arma::mat Z_n = Z.row(n)+Z.row(n-1);
      arma::mat U_n = (g -> eval(Z_n));
      arma::mat pred = W*U_n;
      arma::rowvec XB = X.row(n-1)*B;
      pred.each_col() += XB.t();
      arma::rowvec ll = exp(X.row(n)*logcol_softmax(pred));
      arma::vec wt = arma::trans(ll/sum(ll));
      loglik.row(i) += log(mean(ll));
      arma::uvec ind = sample_arma(wt);
      Z_n = Z_n.cols(ind);
      U_n = U_n.cols(ind);
      Z.row(n) = Z_n;
      U.row(n) = U_n;
      pred = pred.cols(ind);
      arma::mat pred2 = col_softmax(pred);
      arma::vec dif = arma::trans(X.row(n))-mean(pred2,1);
      arma::vec meanU = mean(U_n,1);
      d_W -= (dif*arma::trans(meanU));
      d_B -= arma::trans(X.row(n-1))*arma::trans(dif);
    }
    W -= lr*d_W;
    B -= lr*d_B;
    prog.increment();
  }
  return List::create(_["Z"]=Z,_["U"]=U,_["W"]=W,_["B"]=B,_["loglik"]=loglik);
}

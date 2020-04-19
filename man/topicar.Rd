\name{topic_ar}
\alias{topic_ar}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
learn topic ar model
}
\description{
learn topic ar model
}
\usage{
topic_ar(X, sigma, L, n_p, iter, lr, actfun)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
  one-hot encoded document
}
  \item{sigma}{
  standard deviation for the state
}
  \item{L}{
  number of topics
}
  \item{n_p}{
  number of particles
}
  \item{iter}{
  number of iterations
}
  \item{lr}{
  learning rate
}
  \item{actfun}{
  activation function. "softmax", "logistic", "softplus", "sigmoid", or "identity"
  }
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{Z}{state}
\item{U}{topic}
\item{W}{weight matrix for the topic}
\item{B}{weight matrix for the previous word}
\item{loglik}{log likelihood}
%% ...
}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{
Ko Abe
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
out <- topic_ar(X, L=L, sigma=1, n_p=1000, lr=1, iter=50, actfun = "logistic")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

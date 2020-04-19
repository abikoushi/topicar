\name{make_poem}
\alias{make_poem}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
make a poem
}
%%\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
%%}
\usage{
make_poem(U, W, B, vocab, seed = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{U}{
  topic matrix
}
  \item{W}{
  weight matrix for the topic
}
  \item{B}{
  weight matrix for the previous word
}
  \item{vocab}{
  words
}
  \item{seed}{
  seed
}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{poem}{poem}
  \item{seed}{seed}
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
make_poem(U=U,W=out$W,B=out$B,vocab=vocab,seed = 1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

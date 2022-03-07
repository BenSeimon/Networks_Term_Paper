CovFn <- function(MomentMatrix){
# Calculate the variance-covariance matrix
# Moment matrix is N x M, where N is # obs and M is # moments

N <- size(MomentMatrix,1)# number of observations

CovFn <- MomentMatrix'*MomentMatrix
CovFn <- CovFn/N

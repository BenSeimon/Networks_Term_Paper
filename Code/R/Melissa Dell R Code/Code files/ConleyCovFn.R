ConleyCovFn <- function(MomentMatrix,nei_id,KMN){
# Calculate the variance-covariance matrix

N <- size(MomentMatrix,1)# number of observations
NG <- size(MomentMatrix,2)# number of moment conditions (# m(x_i) )
S <- MomentMatrix; S <- [S; zeros(1,NG)]# moment matrix with extra row of zeros (for 2457 missing value code)
CovFn <- zeros(NG,NG)# # moment x # moment covariance matrix
for (i in 1:N){
    t1 <- MomentMatrix(i,:)# x_i 's for a given observation i; size is # obs x # moments
    t2 <- nei_id(i,:)# matrix of municipalities within 100 km of municipality i
    t3 <-  sum(S(t2,:))*KMN + t1# # obs by # moment size matrix adding up x_i's for each moment condition across all muns within
        # 250 km of mun i, including mun i
    CovFn <- CovFn + t1'*t3# fill in the covariance matrix
}
CovFn <- CovFn/N

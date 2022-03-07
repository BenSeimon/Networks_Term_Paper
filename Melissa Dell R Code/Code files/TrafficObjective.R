TrafficObjective <- function(Parameters,Network, OriginList, MunLinks, MunIds, MunData, Detours){
# Parameters ={delta, gamma, phi_i, phi_p, phi_t, kappa}
# MunData = ValueConf, Interior, tbcx, port, dist,
#           ShContMun, ShLanesMun, ContMun, LanesMun

    # Show current iterate values
    Parameters

    # optimal flows for a given parameter matrix
     #Parameters
    [MunFlows] <- FrankWolfe(Network, Parameters, OriginList, MunLinks, MunIds)
    Flows <- MunFlows(:, 2).*(Parameters(4)./100)
    size(Flows)
    size(MunData)
    MunNum <- [1:length(Flows)]'#InegiCode=MunFlows(:, 1);

    # Merge data
    MergedData <- [zeros(length(MunData), 1), MunData]
    MergedData(MunNum(:, 1), 1) <- Flows(MunNum(:, 1), 1)
    Crossing <- MunData(:, 3) + MunData(:, 4)

    # Calculate terms needed for moment conditions
    Diff <- MergedData(:, 1) - MergedData(:, 2)){# Difference between confiscations and predicted confiscations
    Dist_m <- MergedData(:, 6);      Dist_m <- Dist_m./1e5
    Size <- MergedData(:, 10)+MergedData(:, 9)
    N <- length(Diff)){

    # Mean
    MmeanI <- sum(Diff.*MergedData(:, 3))./N){# mean(interior confiscations) = mean(kappa.* interior flows)
    MmeanT <- sum(Diff.*MergedData(:, 4))./N){# mean(tbcx confiscations) = mean(kappa.* tbcx flows)
    MmeanP <- sum(Diff.*MergedData(:, 5))./N){# mean(port confiscations) = mean(kappa.* port flows)

    # Flows x size
    MsizeT <- sum((Diff.*MergedData(:, 4)).*Size)./N){# mean(tbcx confiscations) = mean(kappa.* tbcx flows)
    MsizeP <- sum((Diff.*MergedData(:, 5)).*Size)./N){# mean(port confiscations) = mean(kappa.* port flows)

    # Flows x detour length
    MdetourI <- sum(Diff.*Detours(:, 1))./N){

    # Flows x distance
    MdistI <- sum(Diff.*Dist_m(:, 1))./N){

    # Variances
    MvarI  <- sum((Diff-mean(Diff)).^2.*MergedData(:, 3))./N){
    MvarC  <- sum((Diff-mean(Diff)).^2.*Crossing(:, 1))./(N)){

     # Rescale
    NI <- sum(MunData(:, 2)); NT <- sum(MunData(:, 3)); NP <- sum(MunData(:, 4))
    MmeanI <- MmeanI.*(N./NI); MmeanT <- MmeanT.*(N./NT).*0.1
    MmeanP <- MmeanP.*(N./NP).*0.05
    MsizeP <- MsizeP.*0.1; MdetourI <- MdetourI.*0.1
    MdistI <- MdistI.*10000
    MvarI <- MvarI.*0.01
    MvarC <- MvarC.*0.1

    G <- [MmeanI, MmeanT, MmeanP, MsizeT, MsizeP, MdetourI, MdistI, MvarI, MvarC]

    # The commented out portion is used for the first step of SMM, the
    # optimal weighting matrix is used in the second step
    load ('WeightMatrix', 'CovFn')# Optimal Weighting Matrix
    Moment <- G*CovFn*G'
    Moment <- Moment.*1000
#     W=eye(9);
#     Moment = G*W*G';
#     Moment=Moment.*10

}

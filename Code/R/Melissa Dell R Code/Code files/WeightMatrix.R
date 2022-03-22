clear
clc
format long g){

##############################
### --------------- Step 1: Load Data and Set Parameters
##############################

# Load parameters
load('1stageSim.mat', 'Par1')

# Load municipality x edge adjacency data
load('MunLinkIds')# st_mun_id, mun_num, node1, node2, length [23,679 x 5]

# Load municipality x edge adjacency data
load('MunIds')# Inegi Code [2259 x 1]

# Load network data
load('RoadNet0absx')#network data: node1 node2 dist bcx port tbcx sh_container sh_commlanes
load('Org0')# Origin matrix
Dest <- 13970# Destination matrix

# Confiscations data
load('Conf', 'Conf')# Conf = [MunNum, id_mun, LnValueConf, ConfDummy]; [2456 x 1]
ValueConf <- Conf{3}
ConfDummy <- Conf{4}
MunNum <- Conf{1}#InegiCode=Conf{2}

# Municipality type ids
load('MunType', 'MunType')# MunType = st_mun_id Mun_num interior tbcx port
interior <- MunType{3}
tbcx  <- MunType{4}
port  <- MunType{5}
MunNum2  <- MunType{2}

# Average length of roads passing through each municipality
load('MunDist', 'MunDist')#st_mun_id Mun_num LinkLength
Dist_m <- MunDist{3}
MunNum3  <- MunDist{2}

# Measured capacity
   # st_mun_id Mun_num commlanes shComm containers ShContainer
load ('MeasuredCap', 'MeasuredCap')# st_mun_id Mun_num sh_container sh_commlanes
MunNum4 <- MeasuredCap{2}
LanesMun <- MeasuredCap{3}
ShLanesMun <- MeasuredCap{4}
ContMun <- MeasuredCap{5}
ShContMun <- MeasuredCap{6}

# Prep confiscations and municipality type data
MunData <- zeros(length(ValueConf), 7)
MunData(MunNum(:, 1), 1) <- ValueConf(MunNum(:, 1), 1)
MunData(MunNum2(:, 1), 2) <- interior(MunNum2(:, 1), 1)
MunData(MunNum2(:, 1), 3) <- tbcx(MunNum2(:, 1), 1)
MunData(MunNum2(:, 1), 4) <- port(MunNum2(:, 1), 1)
MunData(MunNum3(:, 1), 5) <- Dist_m(MunNum3(:, 1), 1)
MunData(MunNum4(:, 1), 6) <- ShContMun(MunNum4(:, 1), 1)
MunData(MunNum4(:, 1), 7) <- ShLanesMun(MunNum4(:, 1), 1)
MunData(MunNum4(:, 1), 8) <- ContMun(MunNum4(:, 1), 1)
MunData(MunNum4(:, 1), 9) <- LanesMun(MunNum4(:, 1), 1)
clear ValueConf tbxc interior port MunType Conf

# Detour Length
load('Centrality.mat', 'PathLength')

LinkId <- [Network{1}, Network{2}; Network{2}, Network{1}]#--Link Identifiers
Distance <- [Network{3}; Network{3}]#---Distance
CostMatrix <- sparse(LinkId(:, 1), LinkId(:, 2), Distance)# Cost matrix
BaseNetwork <- NetworkLength(CostMatrix, OriginList)# Get baseline distance
PathLength <- PathLength-BaseNetwork# Calculate difference

Detours <- zeros(2456,1)# Detours will give detour length if detour possible, and is equal to zero otherwise;
Detours(1:length(PathLength)) <- PathLength(1:length(PathLength))# include municipalities w/o a road, detour equals zero

DetourPossible <-  zeros(2456,1)#  dummy = 0 if blocking mun -> some drugs can't get to border
DetourPossible(Detours!=Inf)=1

Detours(Detours==Inf)=0# code muns where's its not possible to get all drugs to the border without going through that mun
    # as having a detour of zero; these won't be included in  moment conditions
Detours <- Detours./100

Detours <- [Detours, DetourPossible]

########################################
## Step 2: estimation
########################################

MomentMat <- MomentMatrix(Par1,Network, OriginList, MunLinks, MunIds, MunData, Detours)

CovFn <- CovFn(MomentMat)

CovFn <- inv(CovFn)

save('WeightMatrix', 'CovFn')

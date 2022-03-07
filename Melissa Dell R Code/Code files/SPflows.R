SPflows <- function(Network, OriginList, MunLinks, MunIds){

##############################
### --------------- Step -1: Initialize matrices and prep data
##############################

# Create sorted list of cost inputs
LinkId <- [Network{1}, Network{2}; Network{2}, Network{1}]#--Link Identifiers
Distance <- [Network{3}; Network{3}]#---Distance+alpha*PAN

# Create a single index identifier for each link
id <- [1:length(LinkId)]'

# Create a matrix with the inputs for updating the cost function
CostInputs <- [LinkId, Distance, id]
clear LinkId Distance Bcx Cap id Container Lanes

# Create a sparse matrix containing a single index identifier for
# each link
LinkIdMatrix <- sparse(CostInputs(:, 1), CostInputs(:, 2), CostInputs(:, 4))
Distance <- sparse(CostInputs(:, 1), CostInputs(:, 2), CostInputs(:, 3))# Create bi-directonal distance matrice

# Prepare mun inputs
EdgeMunId <- MunInputs(MunLinks, LinkIdMatrix)

# Run Dijkstra's algorithm
[dEM1, dEM2] <- size(EdgeMunId)
size(EdgeMunId)
PathMun <- zeros(length(OriginList), dEM2)
for (j in 1:length(OriginList)){
    [d, pred] <- shortest_paths(Distance, OriginList(j), struct('algname','dijkstra'))
    PathMun <- Path2Mun(OriginList(j), j, pred,LinkIdMatrix, EdgeMunId, PathMun, dEM1)
}

# Aggregate mun flows across paths
MunFlows <- sum(PathMun)
MunFlows <- MunFlows'
size(MunFlows)
MunFlows <- [MunIds{1},  MunFlows]

###################
###------------------------Subfunctions
###################

    function [EdgeMunId] <- MunInputs(MunLinks, LinkIdMatrix)

        # Make mun-edge pairs bi-directional
        #InegiKey=[MunLinks{1}; MunLinks{1}]; # st_mun_id
        MunId <- [MunLinks{2}; MunLinks{2}]; MunNode <- [MunLinks{3}, MunLinks{4}; MunLinks{4}, MunLinks{3}]#bi-directional node and mun ids

        # Merge Mun-edge pairs (currently in mun-node-node form) with the single edge id
        [d1, d2] <- size(LinkIdMatrix)# for creating single index for LinkIdMatrix;
        SingleIndex <- (MunNode(:, 2)- 1).*d1+ MunNode(:, 1)
        EdgeId <- LinkIdMatrix(SingleIndex)
        EdgeId <- full(EdgeId);  MunId <- MunId(EdgeId>0); EdgeId <- EdgeId(EdgeId>0)
        clear d1 d2 SingleIndex MunNode MunLinks
        EdgeMunId <- sparse(EdgeId, MunId, ones(length(MunId), 1), max(EdgeId), 2259)

    }


    function [PathMun] <- Path2Mun(OriginList, j, pred,LinkIdMatrix, EdgeMunId, PathMun, dEM1)

        # Initialize inputs
        LinkId_j <- zeros(500,1)
        nlid <- 0
        i <- 13970

        # Make a list of edges in the optimal path from OriginList(j) - converts lists of nodes traversed into
        # a list of edges traversed
        if (pred(i)!=0){

            while (i ! in  OriginList){
                nlid <- nlid+1
                len <- length(LinkId_j)
                if (nlid>len){
                    LinkId_j(2*len) <- 0
                }
                LinkId_j(nlid, 1) <- LinkIdMatrix(pred(i), i)
                i <- pred(i)
            }
            LinkId_j <- LinkId_j(1:nlid, 1)# list of edges in each path

               # Convert list of edges in the path into a list of Muns in the path
            [dPM1, dPM2] <- size(PathMun)# This is the matrix that we need to fill in
            M <- [1:dPM2]'# Column index for PathMun; max(M)==dMP2
            #pxm_Index=(M-ones(dPM2, 1)).*dPM1+ones(dPM2, 1).*j;  # Linear index for filling in jth row of Path Muns

            PathMunJ <- zeros(dPM2, 1)# Fill in a single row of PathMun so can use PathMun(PathMun>1)=1
            for (Lj in 1:length(LinkId_j)){
                exm_Index <- (M-ones(dPM2, 1)).*dEM1+LinkId_j(Lj).*ones(dPM2, 1)
                PathMunJ <- EdgeMunId(exm_Index)+PathMunJ
            }
            PathMunJ(PathMunJ>1) <- 1# Don't double count paths with multiple edges in same mun
            PathMun(j, :) <- PathMunJ'
        }
    }


}


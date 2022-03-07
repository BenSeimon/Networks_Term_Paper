NetworkLength <- function(CostMatrix, OriginList){

    PathLength <- zeros(length(OriginList), 1)
    for (j in 1:length(OriginList)){
        [d pred] <- shortest_paths(CostMatrix, OriginList(j), struct('algname','dijkstra'))
        PathLength(j, 1) <- d(13970)
    }
    PathLength <- sum(PathLength)



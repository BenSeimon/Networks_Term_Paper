clear
clc
format long g){

### First load data that doesn't change by time period

# Load parameters
load('2stageSim.mat', 'Par1')
Par2 <- Par1

# Load municipality x edge adjacency data
load('MunLinkIds')# st_mun_id, mun_num, node1, node2, length [23,679 x 5]

# Load municipality x edge adjacency data
load('MunIds')# Inegi Code [2259 x 1]

# Load network data

for (i in 12:18){
    i
    load(strcat('RoadNet', int2str(i)))#network data: node1 node2 dist bcx port tbcx sh_container sh_commlanes
    load(strcat('Org', int2str(i)))# Origin matrix

    MunFlows <- FrankWolfe(Network, Par2, OriginList, MunLinks, MunIds)
    csvwrite(strcat('OptFlows', int2str(i), '.csv'), MunFlows)
#
}

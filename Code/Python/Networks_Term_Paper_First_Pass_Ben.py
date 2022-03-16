#!/usr/bin/env python
# coding: utf-8

# # Setup: Install and load packages

# In[21]:


#install packages if required - can also load from the Pipfile
#%pip install networkx


# In[35]:


#import packages 
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import scipy.integrate as integrate 
from scipy.optimize import minimize_scalar
from scipy.misc import derivative
import os


# In[75]:


network_analysis_path_ben = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Networks_Term_Paper/pytrans_UrbanNetworkAnalysis/pytrans/UrbanNetworkAnalysis'
#network_analysis_path_niamh = r'C:\Users\35387\OneDrive\Documents\Networks\New folder\Networks_Term_Paper\pytrans_UrbanNetworkAnalysis\pytrans\UrbanNetworkAnalysis'
os.chdir(network_analysis_path_ben)
import Frank_Wolfe
import TransportationNetworks as tn
import visualize_graph


# # Read in data

# In[ ]:


#data sourced from https://github.com/bstabler/TransportationNetworks


# In[3]:


#set path parameters
data_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Networks_Term_Paper/Data/'
data_path
city = 'Birmingham'
network_file = city+'_net.tntp'
node_file = city+'_nodes.tntp'
trip_file = city+'_trips.tntp'


# In[4]:


#below code taken from https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb

#load network file
netfile = data_path + city + '/' + network_file
brum = pd.read_csv(netfile, skiprows=8, sep='\t')
trimmed= [s.strip().lower() for s in brum.columns]
brum.columns = trimmed
# And drop the silly first andlast columns
brum.drop(['~', ';'], axis=1, inplace=True)

#load node file - note this is a bit fiddly, could be a better way to import but chose something quick and dirty
nodefile = data_path + city + '/' + node_file
brum_nodes = pd.read_csv(nodefile, sep = ' ')
for_drop = []
for i in list(range(1,14)):
    name = 'Unnamed: ' + str(i)
    for_drop.append(name)
for_drop.remove('Unnamed: 7')
brum_nodes = brum_nodes.drop(for_drop, axis = 1)


# In[5]:


brum_nodes


# In[6]:


brum.head()


# In[59]:


print('Total number of nodes:', len(brum['init_node'].unique()))
print('Total number of links:', brum.shape[0])


# To note:
# 
# - No idea what b is 
# - Below is a couple of definitions from the GitHub page which might come in handy
# 
# Link travel time = free flow time * ( 1 + B * (flow/capacity)^Power ).
# Link generalized cost = Link travel time + toll_factor * toll + distance_factor * distance

# # plot graph

# In[11]:


#take a small subset of the graph - 
brum_subset = brum.head(round(brum.shape[0]*0.005))


# In[12]:


print('Total number of nodes:', len(brum_subset['init_node'].unique()))
print('Total number of links:', brum_subset.shape[0])


# In[27]:


brum_subset_graph = nx.from_pandas_edgelist(brum_subset, source = 'init_node',target='term_node', edge_attr = True)


# In[55]:


all_subset_nodes = brum_subset['init_node'] + brum_subset['term_node']
brum_subset_nodes = all_subset_nodes.unique()


# In[63]:


brum_subset_nodes = brum_nodes[brum_nodes['NodeID'].isin(list(brum_subset_nodes))]


# In[51]:


def setnodeinfo(G, nodes_df):
    nodes_name = nodes_df["NodeID"].to_numpy()
    nodes_x = nodes_df["Xcoord"].to_numpy()
    nodes_y = nodes_df["Ycoord"].to_numpy()
    pos ={}
    for i in range(len(nodes_name)):
        G.nodes[nodes_name[i]]["loc"]=(nodes_x[i],nodes_y[i])
        pos[nodes_name[i]]=(nodes_x[i],nodes_y[i])
    return pos


# In[64]:


pos = setnodeinfo(brum_subset_graph, brum_subset_nodes)


# In[65]:


pos = setnodeinfo(brum_subset_graph, brum_nodes)
nx.draw_networkx_nodes(G,pos,node_size=200)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()


# # Compute equilibrium flow

# ## below by importing the class

# In[38]:


## Load pytrans library
network_analysis_path_ben = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Networks_Term_Paper/pytrans_UrbanNetworkAnalysis/pytrans/UrbanNetworkAnalysis'
#network_analysis_path_niamh = r'C:\Users\35387\OneDrive\Documents\Networks\New folder\Networks_Term_Paper\pytrans_UrbanNetworkAnalysis\pytrans\UrbanNetworkAnalysis'
os.chdir(network_analysis_path_ben)
import Frank_Wolfe
import TransportationNetworks
import visualize_graph


# In[66]:


#set path
os.chdir(data_path+city)
SO = False
#run Franke_Wolfe algorithm
fw = Frank_Wolfe.Run(network_file, trip_file, node_file, SO)
#fw.showODFlow()
#fw.showODFlowMap()


# ## below replicates the notebook

# ### 1. Functions Defined

# > ***BPR***(_t0, xa, ca, alpha, beta_)
# 
# <ul><ul>
#         : Method for calculating link travel time based on BPR function.<br><br>
#         <u> Parameters </u> :
#         <ul><ul><br>
#                 <li> **t0** _(float)_ - free flow travel time on link a. </li>
#                 <li> **xa** _(float)_ - volume of traffic on line. </li>
#                 <li> **ca** _(float)_ - capacity of link a. </li>
#                 <li> **alpha** _(float)_ - alpha coefficient, usually 0.15 in the BPR curve.</li>
#                 <li> **beta** _(float)_ - beta coefficient, usually 4 in the BPR curve.</li>
#         </ul></ul><br>
#         <u> Returns </u> :
#         <ul><ul><br>
#                 <li> **ta **_(float)_ - travel time for a vehicle on link a.</li>
#         </ul></ul>
# 
# Note this is equivalent to that used by Youn et al. (2008)

# In[69]:


def BPR(t0, xa, ca, alpha, beta):
    ta = t0*(1+alpha*(xa/ca)**beta)
    return ta


# > ***calculateZ***(_theta, network, SO_)
# 
# <ul><ul>
#         : Method for calculating objective function value. <br><br>
#         <u> Parameters </u> :
#         <ul><ul><br>
#                 <li>**theta** _(float)_ - step size which determies how far along the auxility flow the next flow will be.</li>
#                 <li>**network** _(dictionary)_ - graph of the current network. </li>
#                 <li>**SO** _(string)_ - True: if the objective is to find system optimum, False: User equilibrium </li>
#         </ul></ul><br>
#         <u> Returns </u> :
#         <ul><ul><br>
#                 <li>**z** _(float)_ - estimated objective</li>

# In[71]:


def calculateZ(theta, network, SO):
    z = 0
    for linkKey, linkVal in network.items():
        t0 = linkVal['t0']
        ca = linkVal['capa']
        beta = linkVal['beta']
        alpha = linkVal['alpha']
        aux = linkVal['auxiliary'][-1]
        flow = linkVal['flow'][-1]
        
        if SO == False:
            z += integrate.quad(lambda x: BPR(t0, x, ca, alpha, beta), 0, flow+theta*(aux-flow))[0]
        elif SO == True:
            z += list(map(lambda x : x * BPR(t0, x, ca, alpha, beta), [flow+theta*(aux-flow)]))[0]
    return z


# > ***lineSearch***(_network, SO_)
# 
# <ul><ul>
#         : Finds theta, the optimal solution of the line search that minimizing the objective function along the line between current flow and auxiliary flow.<br><br>
#         <u> Parameters </u> :
#         <ul><ul><br>
#                 <li>**network** _(dictionary)_ - graph of the current network. </li>
#                 <li>**SO** _(string)_ - True: if the objective is to find system optimum, False: User equilibrium </li>
#         </ul></ul><br>
#         <u> Returns </u> :
#         <ul><ul><br>
#                 <li>**theta.x** _(float)_ - optimal move size</li>

# In[72]:


def lineSearch(network, SO):
    theta = minimize_scalar(lambda x: calculateZ(x, network, SO), bounds = (0,1), method = 'Bounded')
    return theta.x


# ### Major Variables
# 
# #### network _(dictionary)
# - t0 (float) : cost at the free flow speed
# - capa (float) : capacity of the link
# - alpha (float) : alpha coefficient of the BPR funtion, usually 0.15 - defined as 0.2 by Youn et al
# - beta (float) : the exponent of power of the BPR function, usually 4 - defined as 10 by Youn et al
# - cost (list) : unit cost of the link at the condition of each iteration
# - auxiliary (list) : the auxiliary flow of F-W algorithm at the condition of each iteration
# - flow (list) : the assigned flow at the condition of each iteration
# 
# #### fwResult _(dictionary)
# - theta (list) : step size at each iteration
# - z (list) : total value of the objective function

# ### file paths

# In[73]:


#set path parameters
data_path = '/Users/benseimon/Documents/Barca GSE/Studies/Term 2/Networks/Term Paper/Networks_Term_Paper/Data/'
data_path
city = 'Birmingham'
network_file = city+'_net.tntp'
node_file = city+'_nodes.tntp'
trip_file = city+'_trips.tntp'


# In[80]:


brum_subset_graph.edges()


# In[81]:


for edge in brum_subset_graph.edges():
    snode, enode = edge[0], edge[1]


# In[83]:


enode


# In[86]:


nodeposition


# In[85]:


px1, py1 = nodeposition[snode][0], nodeposition[snode][1]


# In[84]:


nodeposition = nx.get_node_attributes(brum_subset_graph, "pos")


# ### Set objective and Open a network

# In[78]:


os.chdir(data_path+city)
SO = False # True - System optimum, False - User equilibrium
Brum = tn.Network(network_file, trip_file, node_file, SO)


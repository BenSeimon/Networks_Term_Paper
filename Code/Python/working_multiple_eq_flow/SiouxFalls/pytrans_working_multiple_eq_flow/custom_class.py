#!/usr/bin/env python
# coding: utf-8

# In[1]:


import TransportationNetworks as tn
import Frank_Wolfe
import pandas as pd
import networkx as nx

#create class
#for future iterations - redesign so it takes a city as the class
class fw_custom_algorithm():
    
    def __init__(self, cities_dict):
        self.cities_dict = cities_dict
        self.link_fields = {"from": 1, "to": 2, "capacity": 3, "length": 4, "t0": 5, "B": 6, "beta": 7, "V": 8}
    #create a csv for each city and corresponding file for investigation
 
    #need to fix this to make 
    def create_links_csv(self):
        for city in self.cities_dict.keys():
            path_list = list(self.cities_dict[city]['file_paths'])
            link_file = path_list[0]
            csv = pd.read_csv(self.cities_dict[city]['file_paths'][link_file], skiprows=7, sep='\t') #gets link file path
            trimmed= [s.strip().lower() for s in csv.columns]
            csv.columns = trimmed
            # And drop the silly first andlast columns
            csv.drop(['~', ';'], axis=1, inplace=True)
            self.cities_dict[city]['csv'] = csv        
    
    def plot(self, city):
        csv = self.cities_dict[city]['csv']
        network = nx.from_pandas_edgelist(csv, source = 'init node',target='term node', edge_attr = True)
        graph = nx.complete_graph(network)
        print('Graph for:', city)
        nx.draw(graph)
        
    def summary_stats_all(self):
        for city in self.cities_dict.keys():
            origin_nodes = set(self.cities_dict[city]['csv']['init node'].unique())
            destination_nodes = set(self.cities_dict[city]['csv']['term node'].unique())
            unique_nodes = list(origin_nodes | destination_nodes)
            print('Total number of nodes in', city+':', len(unique_nodes))
            print('Total number of edges in', city+':', self.cities_dict[city]['csv'].shape[0])
            
    def make_network(self, city, remove_link_number): #remove_link_number can be None or int
        #instatiate class using required files 
        link_file = self.cities_dict[city]['file_paths']['link_file_path']
        trip_file = self.cities_dict[city]['file_paths']['trip_file_path']
        node_file = self.cities_dict[city]['file_paths']['node_file_path']
        Network = tn.Network(remove_link_number, link_file, trip_file, node_file) 
        self.cities_dict[city][str(remove_link_number)] = {'network': Network}
    
    def make_network_shut_each_link(self, city): 
        #max_edges = 10 #for a trial 
        max_edges =  self.cities_dict[city]['csv'].shape[0] #get max edges in network for a city
        #iterate over edges
        for edge in range(0, max_edges):
            #instatiate class for network with one file removed using required files 
            link_file = self.cities_dict[city]['file_paths']['link_file_path']
            trip_file = self.cities_dict[city]['file_paths']['trip_file_path']
            node_file = self.cities_dict[city]['file_paths']['node_file_path']
            Network = tn.Network(edge, link_file, trip_file, node_file) 
            self.cities_dict[city][str(edge)] = {'network': Network}
            
    def network_attributes(self, city, remove_link_number):
        network = self.cities_dict[city][str(remove_link_number)]['network']
        ##Network has three attributes
        #1) graph object
        graph = network.graph
        #2) origin nodes
        origin_nodes = network.origins
        #3)dict: keys= origin_node, destination_node, value = traffic flow
        flows = network.od_vols
        print(network)
        print(city, 'graph is:', graph)
        print(city, 'origin nodes are:', origin_nodes)
        print(city, 'flows are:', flows)
    
    def compute_link_flow(self, city, remove_link_number):
        network = self.cities_dict[city][str(remove_link_number)]['network']
        SO = False
        fw = Frank_Wolfe.Run(network, SO)
        #saves file to dict
        self.cities_dict[city][str(remove_link_number)]['fw_run'] = fw
        
    def eq_flow_shut_each_link(self, city):
        max_edges = len(self.cities_dict[city]['None']['network'].graph.edges)#get max edges in network for a city
        #max_edges = 10 #for a trial 
        #iterate over edges
        for edge in range(0, max_edges):
            self.compute_link_flow(city, edge)


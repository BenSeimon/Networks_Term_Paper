import networkx as nx
import scipy.integrate as integrate 
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import math
from scipy.misc import derivative

import TransportationNetworks as tn

class Run:
    """
    Class of implementing Frank-Wolfe algorithm for networks privided from 
    Transportation Networks for Research Core Team (https://github.com/bstabler/TransportationNetworks)
    
    Parameters
    ----------
    link_file :     string
                    file path of network file, which containing various link information
                    
    trip_file :     string
                    file path of trip table. An Origin label and then Origin node number, followed by Destination node numders and OD flow
                    
    node_file :     string
                    file path of node file, which containing coordinates information of nodes
                    
    SO:             boolean
                    True if objective is to find system optimal solution,
                    False if objective is to find user equilibrium
                    
    Attributes
    ----------
    graph:          networkx DiGraph
                    graph of links when completing the algorithm
                    
    network:        nested dictionary
                    dictionary of links information and history of Frank-Wolfe algorithm implementation by iteration
    
    fwResult:       dictionary
                    dictionary of theta (optimal move size) and objective function value over iterations
                    
    Example
    -------
    A Quick example
    
    #Set the paths of Transportation Networks file
    
    >>> directory = ".\\Data\\TransportationNetworks\\SiouxFalls\\" 
    >>> link_file = '{}SiouxFalls_net.tntp'.format(directory) 
    >>> trip_file = '{}SiouxFalls_trips.tntp'.format(directory) 
    >>> node_file = '{}SiouxFalls_node.tntp'.format(directory)
    >>> SO = False \n
    
    #Implement Frank-Wolfe algorithm
    
    >>> fw = Run(link_file, trip_file, node_file, SO)
    >>> fw.showODFlow()
    >>> fw.showODFlowMap()
    """
    def __init__(self, link_file, trip_file, node_file, SO, Node, Link, Network):

        """

        :param link_file: Link file - Bargera Network type
        :param trip_file: OD file - Bargera Network type
        :param node_file: Node file - Bargera Network type
        :param SO: Whether the objective function is of System Optimum or not (User Equilibrium)
        """
        self.SO = SO
        
        nw = tn.Network(link_file, trip_file, node_file, self.SO)

        ## initialization
        self.network = {(u,v): {'t0':d['object'].t0, 'alpha':d['object'].alpha, \
                   'beta':d['object'].beta, 'capa':d['object'].capacity, 'flow':[], \
                   'auxiliary':[], 'cost':[]} for (u, v, d) in nw.graph.edges(data=True)}
        
        self.fwResult = {'theta':[], 'z':[]}
        
        nw.all_or_nothing_assignment()
        nw.update_linkcost()
        
        for linkKey, linkVal in self.network.items():
            linkVal['cost'].append(nw.graph[linkKey[0]][linkKey[1]]['weight'])
            linkVal['auxiliary'].append(nw.graph[linkKey[0]][linkKey[1]]['object'].vol)
            linkVal['flow'].append(nw.graph[linkKey[0]][linkKey[1]]['object'].vol)
            
        ## iterations
        iterNum=0
        iteration = True
        while iteration:
            iterNum += 1
            nw.all_or_nothing_assignment()
            nw.update_linkcost()
            
            for linkKey, linkVal in self.network.items():
                linkVal['auxiliary'].append(nw.graph[linkKey[0]][linkKey[1]]['object'].vol)
                
            theta = self.lineSearch()
            self.fwResult['theta'].append(theta)
            
            for linkKey, linkVal in self.network.items():
                aux = linkVal['auxiliary'][-1]
                flow = linkVal['flow'][-1]
                linkVal['flow'].append(flow + theta*(aux-flow))
                
                nw.graph[linkKey[0]][linkKey[1]]['object'].vol =  flow + theta * (aux - flow)
                nw.graph[linkKey[0]][linkKey[1]]['object'].flow = flow + theta * (aux - flow)
                
            
            nw.update_linkcost()
            
            z=0
            for linkKey, linkVal in self.network.items():
                linkVal['cost'].append(nw.graph[linkKey[0]][linkKey[1]]['weight'])
                totalcost = nw.graph[linkKey[0]][linkKey[1]]['object'].get_objective_function()
                z+=totalcost
                
            self.fwResult['z'].append(z)        
            
            if iterNum == 1:
                iteration = True
            else:
                if abs(self.fwResult['z'][-2] - self.fwResult['z'][-1]) <= 0.001 or iterNum==3000:
                    iteration = False
            
        self.graph = nw.graph
                    
    def BPR(self, t0, xa, ca, alpha, beta):
        """
        Method for calculating link travel time based on BPR function
        
        Parameters
        ----------
        t0:     float
                link travel time under free flow speed
                
        xa:     float
                traffic link flow
        
        ca:     float
                capacity of link
                
        alpha:  float
                first BPR function parameter, usually 0.15
                        
        beta:   float
                second BPR function parameter, usually 4.0
                
        Return
        ------
        ta:     float
                link travel time under the current traffic flow
        """
        ta = t0*(1+alpha*(xa/ca)**beta)
        return ta
    
    def calculateZ(self, theta):
        """
        Method for calculating objective function value
        
        Parameters
        ----------
        theta:      float
                    optimal move size
                    
        Return
        ------
        float
            objective function value
                    
        """
        z = 0
        for linkKey, linkVal in self.network.items():
            t0 = linkVal['t0']
            ca = linkVal['capa']
            beta = linkVal['beta']
            alpha = linkVal['alpha']
            aux = linkVal['auxiliary'][-1]
            flow = linkVal['flow'][-1]
            
            if self.SO == False:
                z += integrate.quad(lambda x: self.BPR(t0, x, ca, alpha, beta), 0, flow+theta*(aux-flow))[0]
            elif self.SO == True:
                z += list(map(lambda x : x * self.BPR(t0, x, ca, alpha, beta), [flow+theta*(aux-flow)]))[0]
        return z
    
    def lineSearch(self):
        """
        Method for estimating theta (optimal move size)
        
        Return
        ------
        float
            optimal move size (rate) between 0 and 1
        """
        theta = minimize_scalar(lambda x: self.calculateZ(x), bounds = (0,1), method = 'Bounded')
        return theta.x
    
    def showODFlow(self):
        """
        Method for presenting table of the optimal traffic assignment of the Frank-Wolfe algorithm procedure
        """
        for (u, v, d) in self.graph.edges(data=True):
            print(u, ' -> ', v, ': ', d['object'].vol)
    
    def showODFlowMap(self):
        """
        Method for presenting the traffic assignment result on a map
        """
        edgewidth = [d['object'].vol/5000 for (u, v, d) in self.graph.edges(data=True)]
        plt.figure(num = 1, figsize=(10,10))
        plt.axis('off')
            
        pos = nx.get_node_attributes(self.graph, "pos")
        
        nx.draw_networkx_edges(self.graph, pos, width=edgewidth)
        nx.draw_networkx_edge_labels(self.graph, pos, \
                                         edge_labels={(u, v): round(d["object"].vol,0) for u, v, d in self.graph.edges(data=True)}, \
                                         font_size=8, label_pos=0.3, alpha=0.)
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.show()
            

if __name__ == "__main__":
    directory = ".\\Data\\TransportationNetworks\\SiouxFalls\\"
    link_file = '{}SiouxFalls_net.tntp'.format(directory)
    trip_file = '{}SiouxFalls_trips.tntp'.format(directory)
    node_file = '{}SiouxFalls_node.tntp'.format(directory)
    SO = False
    
    fw = Run(link_file, trip_file, node_file, SO)
    fw.showODFlow()
    fw.showODFlowMap()

######Putting in the Transportation Networks all as a subclass

    class Node:
        """
        Class for handling node object in Transportation Networks

        Parameters
        ----------
        node_id:    int
                    identifier of a node

        """

        def __init__(self, node_id=0):
            self.Node.node_id = node_id


    class Link(object):
        """
        Class for handling link object in Transportation Networks

        Parameters
        ----------
        link_id:    int
                    identifier of link

        length:     float
                    length of link

        capacity:   float
                    capacity of link

        alpha:      float
                    first BPR function parameter, usually 0.15

        beta:       float
                    second BPR function parameter, usually 4.0

        from_node:  int
                    id of origin node of link

        to_node:    int
                    id of destination node of link

        flow:       float
                    flow on link

        free_speed: float
                    free flow speed of link

        v:          float
                    speed limit of link

        SO:         boolean
                    True if objective is to find system optimal solution,
                    False if objective is to find user equilibrium

        Attributes
        ----------
        t0:     float
                link travel time under free flow speed

        time:   float
                link travel time based on the BPR function

        """

        def __init__(self, **kwargs):
            self.link_id = None
            self.length = 0.0
            self.capacity = 0.0
            self.alpha = 0.15 #adapted to align with Youn et al - originally 0.5
            self.beta = 4. #adapted to aligh with Youn et al - originally 4
            self.from_node = 0
            self.to_node = 0
            self.flow = 0.0

            self.free_speed = 1.0
            self._time = None
            self.v = 0.
            self.SO = False

            for k, v in kwargs.items():
                self.__dict__[k] = v

        def get_time(self):
            """
            Method for getting link travel time based on the BPR function \n
            This method is used when setting 'time' variable

            """
            return self.bpr()

        def bpr(self, alpha=None, beta=None, flow=None):
            """
            Method for calculating the BPR function

            Parameters
            ----------
            alpha:      float
                        first BPR function parameter, usually 0.15

            beta:       float
                        second BPR function parameter, usually 4.0

            flow:       float
                        flow on link

            Return
            ------
            float
                link travel time
            """
            if not beta:
                beta = self.Link.beta

            if not flow:
                flow = self.Link.flow
            if not alpha:
                alpha = self.Link.alpha

            try:
                return (self.Link.t0) * (1 + float(alpha) * (float(flow) / float(self.Link.capacity)) ** float(beta))
            except:
                print(flow, self.Link.length, self.Link.free_speed, self.Link.capacity, beta)
                raise
        def gettotalcost_l(self, flow):
            return float(flow)*self.Link.bpr(flow=flow)
        def getmarginalcost_l(self, v):
            return derivative(self.Link.gettotalcost_l, v)
        @property
        def t0(self):
            return float(self.Link.length) / float(self.Link.free_speed)

        @property
        def time(self):
            if self.Link._time:
                return self.Link._time
            return self.Link.get_time()

        def get_objective_function(self):
            """
            Method for calculating objective function value

            Return
            ------
            float
                objective function value        
            """
            return self.Link.t0 * self.Link.flow + self.Link.t0 * self.Link.alpha * math.pow(self.Link.flow, self.Link.beta + 1) / (math.pow(self.Link.capacity, self.Link.beta) * (self.Link.beta + 1))

        def get_bpr_objective_function(self):
            """
            Method for calculating objective function value

            Return
            ------
            float
                objective function value
            """
            return self.Link.t0 * self.Link.flow + self.Link.t0 * self.Link.alpha * math.pow(self.Link.flow, self.Link.beta + 1) / (
            math.pow(self.Link.capacity, self.Link.beta) * (self.Link.beta + 1))

        def get_total_travel_time_function(self):
            """
            Method for calculating objective function value

            Return
            ------
            float
                objective function value
            """
            return self.Link.get_time()*self.Link.flow


    class Network():
        """
        Class for handling Transportation Networks. This class contains methods to read various TNTP format files from the source and methods of network-wide operations

        Parameters
        ----------
        link_file :     string
                        file path of network file, which containing various link information

        trip_file :     string
                        file path of trip table. An Origin label and then Origin node number, followed by Destination node numders and OD flow

        node_file :     string
                        file path of node file, which containing coordinates information of nodes

        SO:             boolean
                        True if objective is to find system optimal solution,
                        False if objective is to find user equilibrium
        Attributes
        ----------
        graph :         networkx.DiGrapy
                        graph of links with Link object and travel time under the current condition

        origins :       list
                        list of origin nodes

        od_vols :       dictionary
                        key: tuple(origin node, destination node), value: traffic flow
        """
        link_fields = {"from": 1, "to": 2, "capacity": 3, "length": 4, "t0": 5, \
                       "B": 6, "beta": 7, "V": 8}

        def __init__(self, link_file, trip_file, node_file=None, SO=False):
            self.Network.link_file = link_file
            self.Network.trip_file = trip_file
            self.Network.node_file = node_file
            self.Network.graph = None
            self.Network.SO = SO
            self.Network.Visualization = self.Network.Visualization()

            self.Network.build_datastructure()

        def build_datastructure(self):
            """
            Method for opening .tntp format network information files and preparing variables for the analysis
            """
            links, nodes = self.Network.open_link_file()
            self.Network.open_trip_file()

            graph = nx.DiGraph()

            for l in links:
                graph.add_edge(l.from_node, l.to_node, object=l, time=l.get_time())

            if self.Network.node_file != None:
                self.Network.open_node_file(graph)
                self.Network.Visualization.reLocateLinks(graph)
            self.Network.graph = graph


        def open_link_file(self):
            """
            Method for opening network file, containing various link information

            Returns
            -------
            list
                list of Link objects having current link condition
            list
                list of Node objects

            """
            f = open(self.Network.link_file)
            lines = f.readlines()
            f.close()

            links_info = []

            header_found = False
            for line in lines:
                if not header_found and line.startswith("~"):
                    header_found = True
                elif header_found:
                    links_info.append(line)

            nodes = {}
            links = []


            for line in links_info:
                data = line.split("\t")

                try:
                    origin_node = str(int(data[self.link_fields["from"]]))
                except IndexError:
                    continue
                to_node = str(int(data[self.Network.link_fields["to"]]))
                capacity = float(data[self.Network.link_fields["capacity"]])
                length = float(data[self.Network.link_fields["length"]])
                alpha = float(data[self.Network.link_fields["B"]])
                beta = float(data[self.Network.link_fields["beta"]])

                if origin_node not in nodes:
                    n = Node(node_id=origin_node)
                    nodes[origin_node] = n

                if to_node not in nodes:
                    n = Node(node_id=to_node)
                    nodes[to_node] = n

                l = Link(link_id=len(links), length=length, capacity=capacity, alpha=alpha, beta=beta,
                         from_node=origin_node, to_node=to_node, flow=float(0.0), SO=self.Network.SO)

                links.append(l)
            return links, nodes.values()

        def open_node_file(self, graph):
            """
            Method for opening node file, containing position information of nodes \n
            This method adds 'pos' key-value pair in graph variable
            """
            f = open(self.Network.node_file)
            n = 0
            for i in f:
                row = i.split("	")
                if n == 0:
                    n += 1
                else:
                    ind, x, y = row[0], float(row[1]), float(row[2])
                    graph.nodes[ind]["pos"] = (x, y)
            f.close()

        def open_trip_file(self, demand_factor=1.0):
            """
            Method for opening trip tables containing OD flows of each OD pair

            Parameter
            ---------
            demand_factor   float
                            demand factor
            """
            f = open(self.Network.trip_file)
            lines = f.readlines()
            f.close()

            self.Network.od_vols = {}
            current_origin = None

            for line in lines:
                if current_origin == None and line.startswith("Origin"):
                    origin = str(int(line.split("Origin")[1]))
                    current_origin = origin

                elif current_origin != None and len(line) < 3:
                    # print "blank",line,
                    current_origin = None

                elif current_origin != None:
                    to_process = line[0:-2]
                    for el in to_process.split(";"):
                        try:
                            dest = str(int(el.split(":")[0]))
                            demand = float(el.split(":")[1]) * demand_factor
                            self.od_vols[current_origin, dest] = demand
                        except:
                            continue
            origins = [str(i) for i, j in self.Network.od_vols]
            self.Network.origins = list(dict.fromkeys(origins).keys())

            od_dic = {}
            for (origin, destination) in self.Network.od_vols:
                if origin not in od_dic:
                    od_dic[origin] = {}

                od_dic[origin][destination] = self.Network.od_vols[origin, destination]
            self.Network.od_dic = od_dic
        def get_od_dic(self):
            return self.Network.od_dic
        def all_or_nothing_assignment(self):
            """
            Method for implementing all-or-nothing assignment based on the current graph. \n
            It updates link traffic flow
            """
            for edge in self.Network.graph.edges(data=True):
                edge[2]['object'].vol = 0

            shortestpath_graph = {}
            for i in self.Network.origins:
                shortestpath_graph[i] = nx.single_source_dijkstra(self.Network.graph, i, weight="weight")
            for (i, j) in self.Network.od_vols:
                odvol = self.Network.od_vols[(i, j)]
                path = shortestpath_graph[str(i)][1][str(j)]
                for p in range(len(path) - 1):
                    fnode, tnode = path[p], path[p + 1]
                    self.Network.graph[fnode][tnode]["object"].vol += odvol

        def update_linkcost(self):
            """
            Method for updating link travel time.
            """
            for (u, v, d) in self.Network.graph.edges(data=True):
                self.Network.graph[u][v]["weight"] = d["object"].time


        class Visualization():
            """
            Class for handling visualization effect
            """
            def reLocateLinks(self, graph):
                """
                Method for modifying links in graph

                Parameter
                ---------
                graph:  networkx DiGraph
                        graph to present
                """
                nodeposition = nx.get_node_attributes(graph, "pos")
                for edge in graph.edges():
                    snode, enode = edge[0], edge[1]
                    px1, py1 = nodeposition[snode][0], nodeposition[snode][1]
                    px2, py2 = nodeposition[enode][0], nodeposition[enode][1]
                    fx, fy, tx, ty = self.Network.Visualization.reLocateAlink(px1, py1, px2, py2)
                    graph[snode][enode]["pos_fnode"] = (fx, fy)
                    graph[snode][enode]["pos_tnode"] = (tx, ty)

            def reLocateAlink(self, px1, py1, px2, py2):
                """
                Method for adjusting location of a link

                Parameters
                ----------
                px1:    float
                        x coordinate of a node

                py1:    float
                        y coordinate of a node

                px2:    float
                        x coordinate of another node

                py2:    float
                        y coordinate of another node
                Returns
                -------
                fx:     float
                        new coordinate of px1

                fy:     float
                        new coordinate of py1

                tx:     float
                        new coordinate of px2

                ty:     float
                        new coordinate of py2
                """
                offset = 5000
                x1, y1 = float(px1), float(py1)
                x2, y2 = float(px2), float(py2)
                dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
                dist = abs(dist ** 0.5)
                sin = (y2 - y1) / dist
                cos = (x2 - x1) / dist
                if x2 - x1 != 0:
                    tan = (y2 - y1) / (x2 - x1)
                else:
                    tan = 1
                if abs(tan) >= 1:
                    fx, fy = x1 + offset * sin, y1 - offset * cos
                    tx, ty = x2 + offset * sin, y2 - offset * cos
                else:
                    fx, fy = x1 + offset * sin, y1 - offset * cos
                    tx, ty = x2 + offset * sin, y2 - offset * cos
                return fx, fy, tx, ty


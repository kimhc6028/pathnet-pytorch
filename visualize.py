import random
import pylab
from matplotlib.pyplot import pause
import networkx as nx
import numpy as np


class GraphVisualize():

    def __init__(self, modules, vis):
        pylab.style.use('ggplot')

        self.node_size_add = 1.5
        self.init_node_size = 0.1
        self.edge_weight_add = 0.1
        self.init_edge_weight = 0.0
        self.fixed_path = [[None] * 3] * 3
        self.fixed_color = None
        self.fixed_weight = 6.4
        pylab.ion()
        self.graph = nx.Graph()
        self.node_ids = {}
        node_num = 0

        self.vis = vis
        if not self.vis:
            print("visualizing graph disabled!!")

        for layer_num, one_layer in enumerate(modules):
            for module_num in range(one_layer):
                self.graph.add_node(node_num, Position=(10 * layer_num, 10 * module_num), size = self.init_node_size)
                self.node_ids[(layer_num, module_num)] = node_num
                node_num += 1

        pylab.show()

    def set_fixed(self, path, color):
        self.fixed_color = color
        self.fixed_path = []
        for layer_num, layer in enumerate(path):
            layer_path = []
            for num in layer:
                layer_path.append(self.node_ids[(layer_num, num)])
            self.fixed_path.append(layer_path)


    def get_fig(self, genes, e_color):

        fixed_pair = [(self.fixed_path[i], self.fixed_path[i+1]) 
                      for i in range(len(self.fixed_path) - 1)]

        for gene in genes:
            gene_pair = [(gene[i], gene[i+1]) for i in range(len(gene) - 1)]

            for layer_num, (pair, fixed) in enumerate(zip(gene_pair, fixed_pair)):
                for first_num in pair[0]:
                    for second_num in pair[1]:
                        first_node = self.node_ids[(layer_num, first_num)]
                        second_node = self.node_ids[(layer_num + 1, second_num)]
                        if self.graph.has_edge(first_node, second_node):
                            self.node_upsize(first_node)
                            self.node_upsize(second_node)
                            weight =  self.graph.get_edge_data(first_node, second_node)['weight']
                            weight += self.edge_weight_add
                            self.graph.add_edge(first_node, second_node, color = e_color, weight = weight)
                        else:
                            self.graph.add_edge(first_node, second_node, color = e_color, weight = self.init_edge_weight)
                        
        for fixed in fixed_pair:
            for f_1 in fixed[0]:
                for f_2 in fixed[1]:
                    if (not f_1 == None) and (not f_2 == None):
                        self.graph.add_edge(f_1, f_2, color = self.fixed_color, weight = self.fixed_weight)

        nodes = self.graph.nodes(data = True)
        node_color = 'g'
        node_size = [node[1]['size'] for node in nodes]
        node_shape = 's'

        edges = self.graph.edges()
        edge_color = [self.graph[u][v]['color'] for u,v in edges]
        weights = [self.graph[u][v]['weight'] for u,v in edges]
        nx.draw_networkx_nodes(self.graph, nodes = nodes, pos=nx.get_node_attributes(self.graph,'Position'), node_color = node_color, node_size = node_size, node_shape = node_shape)
        nx.draw_networkx_edges(self.graph, edges = edges, pos=nx.get_node_attributes(self.graph,'Position'), edge_color = edge_color, width = weights)

    def show(self, genes, color):
        if self.vis:
            self.get_fig(genes, color)
            pylab.draw()
            pause(0.05)
            pylab.clf()
            self.reset()

    def node_upsize(self, node_id):
        size = self.graph.node[node_id]['size']
        size += self.node_size_add
        self.graph.node[node_id]['size'] = size

    def reset(self):
        nodes = self.graph.nodes(data = True)
        edges = self.graph.edges()
        for node in nodes:
            node[1]['size'] = self.init_node_size
        for edge in edges:
            self.graph.remove_edge(*edge)

import random
import pylab
from matplotlib.pyplot import pause
import networkx as nx
import numpy as np


class GraphVisualize():

    def __init__(self, modules):
        self.node_size_add = 1
        self.init_size = 10
        self.weight_size_add = 0.1
        self.init_weight = 0.1

        pylab.ion()
        self.graph = nx.Graph()

        self.node_ids = {}
        node_num = 0

        for layer_num, one_layer in enumerate(modules):
            for module_num in range(one_layer):
                self.graph.add_node(node_num, Position=(10 * layer_num, 10 * module_num), size = self.init_size)
                self.node_ids[(layer_num, module_num)] = node_num
                node_num += 1

        pylab.show()

    def get_fig(self, genes):
        for gene in genes:
            gene_pair = [(gene[i], gene[i+1]) for i in range(len(gene) - 1)]
            for layer_num, pair in enumerate(gene_pair):
                for first_num in pair[0]:
                    for second_num in pair[1]:
                        first_node = self.node_ids[(layer_num, first_num)]
                        second_node = self.node_ids[(layer_num + 1, second_num)]
                
                    if self.graph.has_edge(first_node, second_node):
                        self.node_upsize(first_node)
                        self.node_upsize(second_node)

                        weight =  self.graph.get_edge_data(first_node, second_node)['weight']
                        weight += self.weight_size_add
                        self.graph.add_edge(first_node, second_node, color = 'm', weight = weight)
                    else:
                        self.graph.add_edge(first_node, second_node, color = 'm', weight = self.init_weight)

        nodes = self.graph.nodes(data = True)
        node_color = 'g'
        node_size = [node[1]['size'] for node in nodes]
        node_shape = 's'

        edges = self.graph.edges()
        edge_color = [self.graph[u][v]['color'] for u,v in edges]
        weights = [self.graph[u][v]['weight'] for u,v in edges]
        nx.draw_networkx_nodes(self.graph, nodes = nodes, pos=nx.get_node_attributes(self.graph,'Position'), node_color = node_color, node_size = node_size, node_shape = node_shape)
        nx.draw_networkx_edges(self.graph, edges = edges, pos=nx.get_node_attributes(self.graph,'Position'), edge_color = edge_color, width = weights)
        #return fig

    def show(self, genes):
        self.get_fig(genes)
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
            node[1]['size'] = self.init_size
        for edge in edges:
            self.graph.remove_edge(*edge)

'''        
graph = GraphVisualize([10,10,10])

genes = [np.random.randint(0,10,(3,3)) for _ in range(64)]
a = 0
graph.show(genes)

for i in range(100000000):
    a += i
genes = [np.random.randint(0,10,(3,3)) for _ in range(64)]
graph.show(genes)
for i in range(100000000):
    a += i
genes = [np.random.randint(0,10,(3,3)) for _ in range(64)]
graph.show(genes)
for i in range(100000000):
    a += i
genes = [np.random.randint(0,10,(3,3)) for _ in range(64)]
graph.show(genes)
for i in range(100000000):
    a += i
genes = [np.random.randint(0,10,(3,3)) for _ in range(64)]
graph.show(genes)
for i in range(100000000):
    a += i
'''

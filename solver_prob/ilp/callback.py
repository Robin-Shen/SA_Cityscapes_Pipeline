#!/usr/bin/env python
# coding: utf-8

import cplex
from cplex.callbacks import LazyConstraintCallback
import networkx as nx
from matplotlib import pyplot as plt

class connectivityCallback(LazyConstraintCallback):

    def __call__(self):
        """
        check conncetivity
        if not, add constraints to enforce connectivity
        """
        # build graph
        graph = self._get_graph()

        # check connectivity for each label
        for l in self._label_map:
            # get label and root nodes
            label = self._label_map[l].label
            root = self._label_map[l].root
            #if "background" in label:
            #    continue
            # collect nodes in same label
            nodes = []
            for i in graph.nodes:
                if graph.nodes[i]["label"] == label:
                    nodes.append(i)
            # build subgraph
            sub_graph = graph.subgraph(nodes)
            # check connectivity
            if not nx.is_connected(sub_graph):
                # print("MIP violates connectivity.")
                #print("Adding connectivity constraints for {}...".format(label))
                # get vertex seperations
                seperations, comp_roots = self._k_nearest(graph, nodes, root)
                # get variable names
                for seperation in seperations:
                    for seg in comp_roots:
                        #for i in seperation:
                        #    names.append("x_" + str(i) + "_" + str(l))
                        #names.append("x_" + str(seg) + "_" + str(l))
                        # get variable coefficents
                        #coefs = [1] * len(seperation) + [-1]
                        # add constraints
                        #self.add(constraint = cplex.SparsePair(ind=names, val=coefs), sense = "G", rhs = 0)
                        for k in self._label_map:
                            names = []
                            for i in seperation:
                                names.append("x_{}_{}".format(i, k))
                            names.append("x_{}_{}".format(seg, k))
                            # get variable coefficents
                            coefs = [1] * len(seperation) + [-1]
                            # add constraints
                            self.add(constraint = cplex.SparsePair(ind=names, val=coefs), sense = "G", rhs = 0)
            else:
                #print("{} is connected!".format(label))
                pass


    def _get_graph(self):
        """
        get labelled graph from current feasible solution
        """
        graph = self._graph.copy()
        values = self.get_values()[:len(graph)*len(self._label_map)]
        obj = self.get_objective_value()

        for i in range(len(values)):
            name = self._names[i]
            value = values[i]
            # make assignment
            if value:
                # get i and l
                _, i, l = name.split('_')
                i = int(i)
                l = int(l)
                # get label name
                label = self._label_map[l].label
                # label graph
                graph.nodes[i]["label"] = label

        return graph


    def _k_nearest(self, graph, nodes, root):
        """
        get vertex seperations by k-nearest bfs
        """
        # get sub graph of current label
        sub_graph = graph.subgraph(nodes)

        segs = []
        comp_roots = []
        # contract connected nodes
        for comp in nx.connected_components(sub_graph):
            comp = list(comp)
            # contract root
            if root in comp:
                k = len(comp)
                for node in comp:
                    # skip self
                    if node == root:
                        continue
                    graph = nx.contracted_nodes(graph, root, node)
            # record segs
            else:
                segs += comp
                comp_roots.append(comp[0])

        seperations = []
        # connect root and segs
        queue = [root]
        visited = {root}
        # terminate on k iteration
        for l in range(k):
            # bfs
            new_queue = []
            terminate = False
            for i in queue:
                for n in graph.neighbors(i):
                    if n not in visited:
                        new_queue.append(n)
                        visited.add(n)
                        # terminate when reach seg
                        if n in segs:
                            comp_roots.append(n)
                            terminate = True
            queue = new_queue
            if terminate:
                break
            # choose odd distance (even index)
            if l % 2 == 0:
                seperations.append(queue)

        return seperations, comp_roots

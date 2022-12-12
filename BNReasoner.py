from typing import Union
from BayesNet import BayesNet
import pandas as pd
import helper

import networkx as nx
import matplotlib.pyplot as plt


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # UTIL FUNCTIONS ----------------------------------------------------------------------------------------------------

    def find_leaf_nodes(self):
        '''
        A leaf node is a node without children
        returns: list of nodes which are leaf nodes
        '''

        leaf_nodes = []

        for node in self.bn.structure.nodes:
            children = self.bn.get_children(node)
            if len(children) == 0:
                leaf_nodes.append(node)

        return leaf_nodes

    def is_connected(self, start, end):
        '''
        BFS to find if there is a path from start to end
        returns true if nodes start and end are connected
        goes through all connections of all nodes and terminates as soon as a node == endnode. If this never happens before the queue is empty,
        this means that there is no conection

        input: two nodes: start and end
        output: boolean
        '''

        # initialize queue and visited list
        visited= []
        queue = []
        queue.append(start)

        while queue:

            n = queue.pop(0)

            if n == end:
                return True

            if n not in visited:
                visited.append(n)
                neighbours = self.bn.get_children(n) + self.bn.get_parents(n)

                # add neighbours to queue
                # no list comprehension bc that would just create an unneccesary list
                for neighbour in neighbours:
                    queue.append(neighbour)

        # if graph is exhausted, and no connections were found, return False
        return False
        
    # ALGORITHM FUNCTIONS -----------------------------------------------------------------------------------------------

    def node_prune(self, Q, e):
        '''
        prunes all leaf nodes that are not in Q or e
        '''

        leaf_nodes = self.find_leaf_nodes()
        print(leaf_nodes)

        for leaf_node in leaf_nodes:
            if leaf_node not in Q and leaf_node not in e:

                self.bn.del_var(leaf_node)


    def edge_prune(self, Q, e):
        '''
        Prunes the edges of the network, given query variables Q and evidence e
        Input:
        - Q: list of query variables
        - e: dictionary of evidence variables with truth values
        Output:
        - None

        WORKS EXCEPT TABLES NEED TO BE UPDATED STILL
        '''
        
        variables_in_evidence = list(e.keys())

        # find which edges to delete
        edges_to_delete = []
        for edge in self.bn.structure.edges:
            if edge[0] in variables_in_evidence:
                edges_to_delete.append(edge)
        
        # do it in two steps to avoid changing the structure while iterating
        for edge in edges_to_delete:
            self.bn.del_edge(edge) 

        # use bm.reduce_factor() to update the tables (not really sure how to do this)
        # or use get_compatible_instantiation table! (someone said this in the zoom)        
        # evidence for tables in which evidence is the given variable

    def network_pruning(self, Q, e):

        self.edge_prune(Q, e)
        self.node_prune(Q, e)

    def d_seperation(self, x: set, y: set, z: set) -> bool:
        '''
        if all paths from x to y are blocked by z, then x and y are d-separated
        if there is a path from x to y that is not blocked by z, then x and y are not d-separated
        Paths between sets of nodes can be exponentially many. By using pruning, d-seperation can be computed in linear time.
        '''

        # do until no new nodes or edges can be deleted:
        while True:

            no_changes = 0

            # delete every leaf node, except those in x, y, or z
            old_nodes = self.bn.get_all_variables()

            leaf_nodes = self.find_leaf_nodes()
            [self.bn.del_var(leaf_node) for leaf_node in leaf_nodes if leaf_node not in x and leaf_node not in y and leaf_node not in z]
            new_nodes = self.bn.get_all_variables()

            # no nodes deleted
            if old_nodes == new_nodes:
                no_changes += 1

            # delete all outgoing edges from nodes in z
            edges_deleted = 0
            for node in z:
                children = self.bn.get_children(node)
                edges_deleted += len(children)
                for child in children:
                    self.bn.del_edge((node, child))
            
            if edges_deleted == 0:
                no_changes += 1

            if no_changes >= 2:
                break

        # if there is a path from x to y that is not blocked by z, then x and y are not d-separated
        # d-seperated iff all possible connections don't have a path
        # return False as soon as a connection is found
        for var_x in x:
            for var_y in y:
                if self.is_connected(var_x, var_y):
                    return False

        # otherwise return True
        return True

    def indepencence(self, x, y, z):    
        '''
        Given three sets of variables x, y, and z, determine wether x and y are independent given z
        input:
        x: set of vars
        y: set of vars
        z: set of vars
        output: 
        bool
        '''

        # IS THIS ALL?!
        return self.d_seperation(x, y, z)

    def sum_out(self, X):
        
        cpt = self.bn.get_cpt(X)
        all_other_vars = [v for v in cpt.columns if v not in ['p', X]]
        new_cpt = cpt.groupby(all_other_vars).sum().reset_index()[all_other_vars + ['p']]
        
        return new_cpt

    def max_out(self, X):
            
        cpt = self.bn.get_cpt(X)
        all_other_vars = [v for v in cpt.columns if v not in ['p', X]] 
        new_cpt = cpt.groupby(all_other_vars).max().reset_index()
        
        return new_cpt
    
    def multiply_factors(self, fact_1, fact_2):

        common_columns = [var for var in bn.bn.get_all_variables() if var in fact_1.columns and var in fact_2.columns]
        new_cpt = pd.merge(fact_1, fact_2, on = common_columns, how='outer')
        new_cpt['p'] = new_cpt['p_x'] * new_cpt['p_y']
        new_cpt.drop(['p_x', 'p_y'], axis=1, inplace=True)
        
        return new_cpt

    def ordering(self, x, strategy=None):
        '''
        Given a set of variables X in the Bayesian network, computes a good ordering for the elimination of X based on:
         - the min-degree heuristics (degree: how many edges the node has in the interaction graph)
         - the min-fill heuristics (order based on least amount of edges that have to be added to keep nodes connected in the interaction graph)

        input: x - set of variables
        strategy: 'min-degree' or 'min-fill'

        returns: list of ordering of those variables
        '''

        order = []
        int_graph = self.bn.get_interaction_graph()

        if strategy == 'min-degree':

            # get degrees
            degree_dict = {}
           
            for var in x:
                degree = len([c[0] if c[0] != var else c[1] for c in int_graph.edges if var in c])
                degree_dict[var] = degree

            # return ordering, sort dict by values and create list of sorted keys
            order = [k for k, v in sorted(degree_dict.items(), key=lambda item: item[1])]

            return order

        elif strategy == 'min-fill':

            fill_dict = {}
            for var in x:
                
                # Extract neighbor nodes from the interaction graph
                node_list = [c[0] if c[0] != var else c[1] for c in int_graph.edges if var in c]
                
                # Iterate over nodes
                needed_edges = 0
                for node_1 in node_list:
                    for node_2 in node_list:
                        if node_2 != node_1:
                            edge = (node_1, node_2)
                            if edge not in int_graph.edges: # For all connected node pairs, check if they are connected already
                                needed_edges += 1 # If not, add 1
                    node_list.remove(node_1) # Remove checked node to avoid double counting
                
                fill_dict[var] = needed_edges

            order = [k for k, v in sorted(fill_dict.items(), key=lambda item: item[1])]

            return order

        else:
            raise Exception('Please specify a ordering strategy, either min-degree or min-fill')


if __name__ == '__main__':
    # Load the BN from the BIFXML file
    reasoner = BNReasoner('testing/dog_problem.bifxml')
    # Reasoner.bn.draw_structure()

    net = reasoner.bn

    net.draw_structure()
    plt.show()

    nx.draw(net.get_interaction_graph(), with_labels=True)
    plt.show()



    # test is_connected()
    # helper.test_function(Reasoner.d_seperation({'bowel-problem'}, {'family-out'}, {'light-on'}))
    helper.test_function(Reasoner.ordering({'dog-out', 'family-out', 'light-on'}, strategy='min-degree'))
    



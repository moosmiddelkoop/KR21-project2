from typing import Union
from BayesNet import BayesNet
import pandas as pd


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

    # TODO: This is where your methods should go

    def find_leaf_nodes(self):
        '''
        A leaf node is a node without children
        returns: list of nodes which are leaf nodes

        DOESNT WORK
        '''

        leaf_nodes = []

        for node in self.bn.structure.nodes:
            children = self.bn.get_children(node)
            if len(children) == 0:
                leaf_nodes.append(node)

        return leaf_nodes

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

    def network_pruning(self, Q, e):

        self.edge_prune(Q, e)
        self.node_prune(Q, e)


if __name__ == '__main__':
    # Load the BN from the BIFXML file
    Reasoner = BNReasoner('testing/dog_problem.bifxml')

    # Print the interaction graph
    Reasoner.bn.draw_structure()
    Reasoner.network_pruning(['hear-bark'], {'dog-out': True})
    Reasoner.bn.draw_structure()




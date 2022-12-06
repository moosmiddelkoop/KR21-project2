from typing import Union
from BayesNet import BayesNet


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

    def node_prune(self, Q, e):



        pass

    def edge_prune(self, Q, e):
        '''
        Prunes the edges of the network, given query variables Q and evidence e
        Input:
        - Q: list of query variables
        - e: dictionary of evidence variables with truth values
        Output:
        - None
        '''

        # find which edges to delete
        edges_to_delete = []
        for edge in self.bn.structure.edges:
            if edge[0] == e:
                edges_to_delete.append(edge)
        
        # do it in two steps to avoid changing the structure while iterating
        for edge in edges_to_delete:
            self.bn.del_edge(edge) 

        # use bm.reduce_factor() to update the tables (not really sure how to do this)
        


    def network_pruning(self, Q, e):

        self.edge_prune(Q, e)
        self.node_prune(Q, e)


if __name__ == '__main__':
    # Load the BN from the BIFXML file
    bn = BNReasoner('testing/dog_problem.bifxml')

    # Print the interaction graph

    bn.edge_prune([], 'hear-bark')




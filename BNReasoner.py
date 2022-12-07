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

    def find_leaf_nodes(self):
        '''
        A leaf node is a node without parents
        returns: list of nodes which are leaf nodes
        '''

        leaf_nodes = []

        for node in self.bn.structure.nodes:
            parents = self.bn.get_parents(node)
            if len(parents) == 0:
                leaf_nodes.append(node)

        return leaf_nodes


    def node_prune(self, Q, e):
        '''
        prunes all leaf nodes that are not in Q or e
        '''

        leaf_nodes = self.find_leaf_nodes()

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

        IS DEEPCOPY NEEDED?
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
        # or use get_compatible_instantiation table! (someone said this in the zoom)
        

    def network_pruning(self, Q, e):

        self.edge_prune(Q, e)
        self.node_prune(Q, e)


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



if __name__ == '__main__':
    # Load the BN from the BIFXML file
    bn = BNReasoner('testing/dog_problem.bifxml')

    # Print the interaction graph

    # bn.edge_prune([], 'hear-bark')
    bn.node_prune([], 'hear-bark')






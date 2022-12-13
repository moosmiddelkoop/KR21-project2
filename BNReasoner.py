from typing import List, Tuple, Dict, Union

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

    def find_cpts_per_var(self):
        '''
        returns a dictionary with for each var, a set of the names of all cpts that contain the var
        '''

        # for each var, find which cpts it is in
        cpt_per_var = {}
        all_cpts = self.bn.get_all_cpts()
        for key, cpt in all_cpts.items():
            for column in cpt:
                if column == 'p':
                    continue
                if column not in cpt_per_var:
                    cpt_per_var[column] = {key}
                else:
                    cpt_per_var[column].add(key)

        return cpt_per_var
     
    def set_evidence(self, evidence):
        '''
        given a pd.Series of evidence, set the evidence in the BN, return updated CPTs
        '''
        updated_cpts = {}
        cpts = self.bn.get_all_cpts()
        for var, cpt in cpts.items():
            # reduce factor
            new_cpt = self.bn.reduce_factor(evidence, cpt)

            # remove zero rows (effecitvely summing out the evidence, but keeping variables for multiplication)
            new_cpt = new_cpt[new_cpt['p'] != 0].reset_index(drop=True)

            # update cpt in BN
            self.bn.update_cpt(var, new_cpt)
            # add to updated_cpts for return
            updated_cpts[var] = new_cpt

        return updated_cpts

        
    # ALGORITHM FUNCTIONS -----------------------------------------------------------------------------------------------

    def node_prune(self, Q, e):
        '''
        prunes all leaf nodes that are not in Q or e

        Return deleted nodes
        '''

        leaf_nodes = self.find_leaf_nodes()
        removed = []
        for leaf_node in leaf_nodes:
            if leaf_node not in Q and leaf_node not in e:

                self.bn.del_var(leaf_node)
                removed.append(leaf_node)
        
        return removed


    def edge_prune(self, Q, e):
        '''
        Prunes the edges of the network, given query variables Q and evidence e
        Input:
        - Q: list of query variables
        - e: dictionary of evidence variables with truth values
        Output:
        - Removed edges

        WORKS EXCEPT TABLES NEED TO BE UPDATED STILL
        '''

        # find which edges to delete
        edges_to_delete = []
        for edge in self.bn.structure.edges:
            if edge[0] in e:
                edges_to_delete.append(edge)
        
        # do it in two steps to avoid changing the structure while iterating
        for edge in edges_to_delete:
            self.bn.del_edge(edge)

        return edges_to_delete

        # use bn.reduce_factor() to update the tables (not really sure how to do this)
        # or use get_compatible_instantiation table! (someone said this in the zoom)        
        # evidence for tables in which evidence is the given variable

    def network_pruning(self, Q, e):
        """
        Prunes the network, given query variables Q and evidence e
        """
        # do until no new nodes or edges can be deleted:
        while self.edge_prune(Q, e) + self.node_prune(Q, e) != []:
            pass

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

    def sum_out(self, factor, X):
        
        """
        Given a factor and variable X, return the CPT of X summed out.
        Input:
            factor: Pandas DataFrame representing the CPT of a factor.
            X: string indicating the variable to be summed out.
        Returns:
            Pandas DataFrame: The CPT summed out by X.
        """
        # edge case where you some out over the whole factor
        if len(factor.columns) == 2:
            # remove the row where p = 0, this leaves a single line CPT with the set value (T/F) for the variable
            return factor[factor['p'] != 0].reset_index(drop=True)
        
        cpt = factor
        if X not in cpt.columns:
            raise ValueError("Variable not in CPT")

        all_other_vars = [v for v in cpt.columns if v not in ['p', X]]
        new_cpt = cpt.groupby(all_other_vars).sum().reset_index()[all_other_vars + ['p']]
        
        return new_cpt

    def max_out(self, X):
        
        """Given variable X (as string), return the maxed-out CPT by X, along with the value of X for which the CPT is maxed out.
        Input:
            X: string indicating the variable to be maxed out.
        Returns:
            Pandas DataFrame: The CPT summed out by X.
            Bool: The value of X for which the CPT has the max probability.
        """
            
        cpt = self.bn.get_cpt(X)
        all_other_vars = [v for v in cpt.columns if v not in ['p', X]] 
        new_cpt = cpt.groupby(all_other_vars).max().reset_index()
        max_instantiation = new_cpt[X].iloc[0]
        new_cpt = new_cpt[all_other_vars + ['p']]
        return new_cpt, max_instantiation
    
    
    def multiply_factors(self, fact_1, fact_2):
        
        """
        Given two factors (as CPTs (Pandas Data Frames)), return the outer product of the two factors with new probabilities (probs of the single factors multiplied).
        Input:
            fact_1: First factor (CPT / Pandas DataFrame)
            fact_2: Second factor (CPT / Pandas DataFrame)
        Returns:
            new_cpt: CPT which displays the outer product of the two factors where the probabilities of the single factors are multiplied.
        """

        common_columns = [var for var in self.bn.get_all_variables() if var in fact_1.columns and var in fact_2.columns]
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

        int_graph = self.bn.get_interaction_graph() # get interaction graph

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
                        if node_2 != node_1: # Exlude self-loops
                            edge = (node_1, node_2) 
                            if edge not in int_graph.edges: # For all connected node pairs, check if they are connected already
                                needed_edges += 1 # If not, add 1
                    node_list.remove(node_1) # Remove checked node to avoid double counting
                
                fill_dict[var] = needed_edges 

            order = [k for k, v in sorted(fill_dict.items(), key=lambda item: item[1])] # Sort based on number of needed edges

            return order

        else:
            raise Exception('Please specify a ordering strategy, either min-degree or min-fill')

    
    def var_elimination(self, x, ordering_strat='min-degree'):
        '''
        given a set of variables x, eliminate variables in the right order
        optional input: ordering strategy: 'min-degree' or 'min-fill'. Standard: 'min-degree'
        returns: resulting factor
        '''

        cpt_per_var = self.find_cpts_per_var()

        # get ordering
        order = self.ordering(x, strategy=ordering_strat)

        for i, var in enumerate(order):

            # get multiplication factor (multiply all factors containing the variable)
            cpts_to_combine = [self.bn.get_cpt(v) for v in cpt_per_var[var]]

            # after the first iteration, there is a summed out factor already from last iteration. add it to the list of factors to combine
            if i != 0:
                cpts_to_combine.append(summed_out)

            multiplication_factor = cpts_to_combine[0]
            for i in range(1, len(cpts_to_combine)):
                multiplication_factor = self.multiply_factors(multiplication_factor, cpts_to_combine[i])

            # sum out variable
            summed_out = self.sum_out(multiplication_factor, var)

        return summed_out

    def marginal_distributions(self, query: List[str], evidence: pd.Series) -> List[str]:
        """
        Given a query and evidence, return the marginal distribution of the query variables.
        Input:
            query: List of variable names.
            evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        Returns:
            marginal_distribution: A dictionary with the query variable names as keys and the corresponding marginal distributions as values.
        """
        # prune BN based on evidence
        self.network_pruning(query, evidence)

        # set evidence
        self.set_evidence(evidence)

        # eliminate query variables
        summed_out = self.var_elimination(query)

        return self.bn.get_all_cpts()


    def mep(self, evidence, strategy="min-fill"):
        
        """
        Given some evidence, return the instantiation that is most probable.
        
        Input:
            evidence: Dictionary with variable names as keys and truth values as values.
            strategy: Strategy to order the variables by; "min-fill" or "min-degree".
        Returns:
            instantiation: A dictionary with all variable names as keys and the truth values for which the probability is maximized.
            probability: The probability of the instantiation.
        """
        
        # Frist: Pruning based on evidence
        cpt_dict = {}
        for var in self.bn.get_all_variables():
            cpt = self.bn.get_cpt(var)
            
            # Pruning based on evidence
            for ev in evidence.keys():
                if ev in cpt.columns:
                    cpt = cpt[cpt[ev] == evidence[ev]]   
                             
            cpt_dict[var] = cpt
        
        # No particular order applied here; Could be improved
        order = Reasoner.bn.get_all_variables()

        # Multiply all CPTs
        for i, var in enumerate(order):
            if i == 0:
                running_cpt = cpt_dict[var]
            else:
                running_cpt = self.multiply_factors(running_cpt, cpt_dict[var])
        
        # Extract that instantiation for which the combined probability is maximized
        instantiation = running_cpt[running_cpt["p"] == max(running_cpt["p"])][Reasoner.bn.get_all_variables()].to_dict('records')[0]
            
        return instantiation, max(running_cpt["p"])
    
    def map(self, Q, evidence, strategy="min-fill"):
        
        ### TBD ###
        
        return None 

if __name__ == '__main__':
    # Load the BN from the BIFXML file
    Reasoner = BNReasoner('testing\lecture_example.BIFXML')
    #Reasoner.bn.draw_structure()

    query=["Slippery Road?"]
    evidence=pd.Series({"Rain?": True})

    Reasoner.network_pruning(query, evidence)

    Reasoner.bn.draw_structure()

    result = Reasoner.marginal_distributions(query, evidence)
    for var, cpt in result.items():
        print(var)
        print(cpt)

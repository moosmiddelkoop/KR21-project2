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

    def find_cpts_for_var(self, var):
        '''
        returns a list of all cpts that contain the var

        BROKEN: does not work after pruning
        '''
        cpts_per_var = {var: self.bn.get_cpt(var)}
        for child in self.bn.get_children(var):
            cpts_per_var[child] = self.bn.get_cpt(child)

        return cpts_per_var

    def set_evidence(self, evidence: pd.Series):
        '''
        given a pd.Series of evidence, set the evidence in the BN, return updated CPTs
        '''
        # get all cpts that contain evidence
        cpts = {}
        for var in evidence.index:
            cpts.update(self.find_cpts_for_var(var))

        # set evidence for cpts
        for var, cpt in cpts.items():
            new_cpt = self.bn.get_compatible_instantiations_table(evidence, cpt)
            self.bn.update_cpt(var, new_cpt)
            
        return cpts

        
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


    def edge_prune(self, Q, e: pd.Series):
        '''
        Prunes the edges of the network, given query variables Q and evidence e
        Input:
        - Q: list of query variables
        - e: pd.Series of evidence variables with truth values {var: val}
        Output:
        - Removed edges
        '''
        removed = []
        # prune all edges that go out of evidence nodes (for each node U in e: del U -> X, update CPT of X)
        for var, val in e.items():
            for child in self.bn.get_children(var):
                edge = (var, child)
                # remove edge (U -> X) from network
                self.bn.del_edge(edge)

                # reduce factor and sum out U, update CPT of X
                reduced_factor = self.bn.reduce_factor(pd.Series({var: val}), self.bn.get_cpt(child))
                reduced_factor = self.sum_out(reduced_factor, var)
                self.bn.update_cpt(child, reduced_factor)

                removed.append(edge)

        return removed


    def network_pruning(self, Q, e):
        """
        Prunes the network, given query variables Q and evidence e
        """
        # do until no new nodes or edges can be deleted:
        while self.edge_prune(Q, e) + self.node_prune(Q, e) != []:
            pass
   
       
    def is_d_seperated(self, X: set, Y: set, Z: set) -> bool:
        '''
        if all paths from x to y are blocked by z, then x and y are d-separated
        if there is a path from x to y that is not blocked by z, then x and y are not d-separated
        Paths between sets of nodes can be exponentially many. By using pruning, d-seperation can be computed in linear time.
        '''

        # we can use network pruning for this if we set Q = x U y, e = z
        Q = X.union(Y)
        e = {z: True for z in Z}  # set mock value for evidence to not frustrate code
        self.network_pruning(Q, e)

        # if there is a path from x to y that is not blocked by z, then x and y are not d-separated
        # d-seperated iff all possible connections don't have a path
        # return False as soon as a connection is found
        for x in X:
            for y in Y:
                if self.is_connected(x, y):
                    return False

        # all paths are disconnected, return True
        return True

    def indepencence(self, X, Y, Z):    
        '''
        Given three sets of variables x, y, and z, determine wether x and y are independent given z
        input:
        x: set of vars
        y: set of vars
        z: set of vars
        output: 
        bool
        '''
        # IS THIS ALL?! #TODO: There is an edge case where odds are fifty fifty, d-seperation would return False even though the var IS indepedent
        return self.is_d_seperated(X, Y, Z)

    def sum_out(self, factor, X):
        """
        Given a factor and variable X, return the CPT of X summed out.
        Input:
            factor: Pandas DataFrame representing the CPT of a factor.
            X: string indicating the variable to be summed out.
        Returns:
            Pandas DataFrame: The CPT summed out by X.
        """
        cpt = factor
        if X not in cpt.columns:
            raise ValueError("Variable not in CPT")

        all_other_vars = [v for v in cpt.columns if v not in ['p', X]]

        # edge case where there are no other variables
        if len(all_other_vars) == 0:
            # sum on p
            return pd.DataFrame({'p': [cpt['p'].sum()]})
        
        # sum out x
        new_cpt = cpt.groupby(all_other_vars).sum().reset_index()[all_other_vars + ['p']]
        
        return new_cpt

    def max_out(self, factor, X):
        
        """Given variable X (as string), return the maxed-out CPT by X, along with the value of X for which the CPT is maxed out.
        Input:
            X: string indicating the variable to be maxed out.
        Returns:
            Pandas DataFrame: The CPT summed out by X.
            Bool: The value of X for which the CPT has the max probability.
        """
                    
        cpt = factor
        if X not in cpt.columns:
            raise ValueError("Variable not in CPT")
        
        all_other_vars = [v for v in cpt.columns if v not in ['p', X]] 
        if len(all_other_vars) == 0:
            return cpt.max()[1], True if cpt.max()[0] == 1.0 else False

        new_cpt = cpt.groupby(all_other_vars).max().reset_index()
        max_instantiation = new_cpt[X].iloc[0]
        new_cpt = new_cpt[all_other_vars + ['p']]
        return new_cpt, max_instantiation
    
    
    def multiply_factors(self, f1, f2):
        """
        Given two factors (as CPTs (Pandas Data Frames)), return the outer product of the two factors with new probabilities (probs of the single factors multiplied).
        Input:
            fact_1: First factor (CPT / Pandas DataFrame)
            fact_2: Second factor (CPT / Pandas DataFrame)
        Returns:
            new_cpt: CPT which displays the outer product of the two factors where the probabilities of the single factors are multiplied.
        """
        common_columns = [var for var in self.bn.get_all_variables() if var in f1.columns and var in f2.columns]

        # edge case if no common columns
        if len(common_columns) == 0:
            new_cpt = f1.merge(f2, how='cross')
        # else merge on common columns
        else:
            new_cpt = pd.merge(f1, f2, on = common_columns, how='outer')
        
        new_cpt['p'] = new_cpt['p_x'] * new_cpt['p_y']
        new_cpt.drop(['p_x', 'p_y'], axis=1, inplace=True)
        
        return new_cpt


    def multiply_cpts_per_var(self, var, cpt_list):
        '''
        Function to multiply all factors that contain a certain variable
        
        input: 
        - var: the variable name as string
        - cpt_list: list of all factors as CPTs (Pandas Data Frames)
        returns:
        - cpts_per_var: list of all factors that contain the variable
        - multiplication_factor: the multiplication of all factors that contain the variable
        '''
        
        # Extract all cpts that contain the variable
        cpts_per_var = [cpt for cpt in cpt_list if var in cpt.columns]
        
        # get multiplication factor (multiply all factors containing the variable)
        multiplication_factor = cpts_per_var[0]
        for j in range(1, len(cpts_per_var)):
            multiplication_factor = self.multiply_factors(multiplication_factor, cpts_per_var[j])
        
        return cpts_per_var, multiplication_factor


    def get_interaction_graph(self, variables):
        """
        Returns a networkx.Graph as interaction graph of the current BN.
        :return: The interaction graph based on the factors of the current BN.
        """
        # Create the graph and add all variables
        int_graph = nx.Graph()
        [int_graph.add_node(var) for var in variables]

        # connect all variables with an edge which are mentioned in a CPT together
        for var in variables:
            involved_vars = list(self.bn.get_cpt(var).columns)[:-1] # Can we use the "normal" CPTs or would we have to use the updated ones?
            for i in range(len(involved_vars)-1):
                for j in range(i+1, len(involved_vars)):
                    if not int_graph.has_edge(involved_vars[i], involved_vars[j]):
                        int_graph.add_edge(involved_vars[i], involved_vars[j])
        return int_graph
    
    
    def ordering(self, x, strategy=None):
        '''
        Given a set of variables X in the Bayesian network, computes a good ordering for the elimination of X based on:
         - the min-degree heuristics (degree: how many edges the node has in the interaction graph)
         - the min-fill heuristics (order based on least amount of edges that have to be added to keep nodes connected in the interaction graph)

        input: x - set of variables
        strategy: 'min-degree' or 'min-fill'

        returns: list of ordering of those variables
        '''

        variables = x.copy() # [i for i in x]
        order = []
        
        if strategy == 'min-degree':
            
            while len(variables) > 0:
                
                int_graph = self.get_interaction_graph(variables) # get interaction graph

                # get degrees
                degree_dict = {}
            
                for var in variables:

                    degree = len([c[0] if c[0] != var else c[1] for c in int_graph.edges if var in c])
                    degree_dict[var] = degree

                # return ordering, sort dict by values and create list of sorted keys
                min_var = [k for k, v in sorted(degree_dict.items(), key=lambda item: item[1])][0]

                order.append(min_var)

                # Now sum-out the variables in the order
                variables.remove(min_var)
            
            return order

        elif strategy == 'min-fill':

            while len(variables) > 0:
                    
                int_graph = self.get_interaction_graph(variables) # get interaction graph

                fill_dict = {}
                for var in variables:
                    
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

                min_var = [k for k, v in sorted(fill_dict.items(), key=lambda item: item[1])][0]

                order.append(min_var)
                
                # Now sum-out the variables in the order
                variables.remove(min_var)
                
            return order

        else:
            raise Exception('Please specify a ordering strategy, either min-degree or min-fill')

    
    def var_elimination(self, x, strategy='min-degree'):
        '''
        given a set of variables x, eliminate variables in the right order
        optional input: ordering strategy: 'min-degree' or 'min-fill'. Standard: 'min-degree'
        returns: resulting factor
        '''

        # get ordering
        order = self.ordering(x, strategy=strategy)

        all_cpts = list(self.bn.get_all_cpts().values())
        
        for var in order:
            
            # Get the multiplication factor and a list of all CPTs that contain the variable
            cpts_per_var, multiplication_factor = self.multiply_cpts_per_var(var, all_cpts)

            # sum out variable
            summed_out = self.sum_out(multiplication_factor, var)
            
            # Delete CPTs of variables that were already summed out
            relevant_cols = [list(cpt.columns) for cpt in cpts_per_var]
            all_cpts = [cpt for cpt in all_cpts if list(cpt.columns) not in relevant_cols]
            
            # Add summed-out CPT, such that it can be multiplied with the remaining CPTs
            all_cpts.append(summed_out)

        return summed_out


    def marginal_distributions(self, query: List[str], evidence: pd.Series, strategy: str) -> List[str]:
        '''
        Given a query and evidence, return the marginal distribution of the query variables.
        Input:
            query: List of variable names.
            evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        Returns:
            marginal_distribution: A dictionary with the query variable names as keys and the corresponding marginal distributions as values.
        '''
        
        # prune BN based on evidence (evidence is now automatically set in the pruning function)
        self.network_pruning(query, evidence)

        # elimate variables not in query
        marginal = self.var_elimination([var for var in self.bn.get_all_variables() if var not in query], strategy = strategy)

        # normalize if evidence is not empty
        if len(evidence) > 0:
            marginal['p'] = marginal['p'] / sum(marginal['p'])

        return marginal


    def mpe(self, evidence, strategy="min-fill"):
        
        """
        Given some evidence, return the instantiation that is most probable.
        
        Input:
            evidence: Dictionary with variable names as keys and truth values as values.
            strategy: Strategy to order the variables by; "min-fill" or "min-degree".
        Returns:
            instantiation: A dictionary with all variable names as keys and the truth values for which the probability is maximized.
            probability: The probability of the instantiation.
        """
        
        self.network_pruning(self.bn.get_all_variables(), evidence)

        var_list = list(self.bn.get_all_cpts().keys())
        
        # Do correct ordering based on all variables
        order = self.ordering(var_list, strategy=strategy)
        
        all_cpts = list(self.bn.get_all_cpts().values())

        for var in order:
            
            cpts_per_var, multiplication_factor = self.multiply_cpts_per_var(var, all_cpts)
            
            # Delete CPTs of variables that were already summed out, but add the combined and summed out CPT instead
            relevant_cols = [list(cpt.columns) for cpt in cpts_per_var]
            all_cpts = [cpt for cpt in all_cpts if list(cpt.columns) not in relevant_cols]
            
            # Add summed-out CPT
            all_cpts.append(multiplication_factor)
            
        # Extract that instantiation for which the combined probability is maximized
        instantiation = multiplication_factor[multiplication_factor["p"] == max(multiplication_factor["p"])][self.bn.get_all_variables()].to_dict('records')[0]
            
        return instantiation, max(multiplication_factor["p"])
    
    
    def map(self, query, evidence, strategy="min-fill"):
        """
        Given a query and some evidence, return the instantiation that maximizes the marginal distribition P(Q/e) along with the probability.
        
        Input:
            query: List of variable names to compute the marginal distribution for, given the evidence.
            evidence: Dictionary with variable names as keys and truth values as values.
            strategy: Strategy to order the variables by; "min-fill" or "min-degree".
        Returns:
            instantiation: A dictionary with all variable names as keys and the truth values for which the probability is maximized.
            probability: The probability of the instantiation.
        """
        
        # Compute marginal distribution
        marginal = self.marginal_distributions(query, evidence, strategy)
        
        # Max out all query variables to get the instantiation for which the probability is maximized
        instantiation = {}
        for var in query:
            maxed_out = self.max_out(marginal, var)
            marginal = maxed_out[0] # Store the new marginal distribution
            instantiation[var] = maxed_out[1] # Store the instantiation for which the probability is maximized
                    
        return instantiation, maxed_out[0] 


if __name__ == '__main__':
    # Load the BN from the BIFXML file
    bnr = BNReasoner('testing\lecture_example.BIFXML')

    bnr.bn.draw_structure()

    query=["Slippery Road?"]
    evidence=pd.Series({"Rain?": False, "Winter?": True})

    bnr.network_pruning(query, evidence)

    bnr.get_interaction_graph()

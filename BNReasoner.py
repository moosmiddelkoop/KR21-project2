from typing import List, Tuple, Dict, Union
from networkx import Graph

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

    def find_cpts_for_var(self, var: str) -> Dict:
        '''
        returns a dict of all cpts that contain the var
        '''
        cpts = {var: self.bn.get_cpt(var)}
        interaction_graph = self.bn.get_interaction_graph()
        for n in interaction_graph.neighbors(var):
            cpt = self.bn.get_cpt(n)
            if var in cpt.columns:
                cpts[n] = self.bn.get_cpt(n)
        
        return cpts


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
            raise ValueError(f"Variable {X} not in CPT")

        all_other_vars = [v for v in cpt.columns if v not in ['p', X]]

        # edge case where there are no other variables
        if len(all_other_vars) == 0:
            # sum on p
            return pd.DataFrame({'p': [cpt['p'].sum()]})
        
        # sum out x
        new_cpt = cpt.groupby(all_other_vars).sum().reset_index()[all_other_vars + ['p']]
        
        return new_cpt

    def max_out(self, factor, X):
        
        """Given variable X (as string), return the maxed-out CPT by X, along with the values of X for which the CPT is maxed out.
        Input:
            X: string indicating the variable to be maxed out.
        Returns:
            Pandas DataFrame: The CPT maxed out by X. The values for X are given after the "p" column.
        """         
        cpt = factor
        if X not in cpt.columns:
            raise ValueError("Variable not in CPT")
        
        # get all other vars up to column p
        all_other_vars = [v for v in cpt.loc[:, :'p'].columns if v not in ['p', X]]
        # edge case where there are no other variables
        if len(all_other_vars) == 0:
            # return row with max p
            new_cpt = cpt[cpt.p == cpt.p.max()]
        else:
            # find max occurrences
            new_cpt = cpt.groupby(all_other_vars)["p"].max()
        
            # this effecively creates an extended factor with the maxed out variable
            new_cpt = pd.merge(cpt, new_cpt, on="p", how="inner").drop_duplicates()
        
        # move all maxed out vars to back to indicate it is set
        new_cpt = new_cpt[[v for v in new_cpt.columns if v != X] + [X]]
        # rename X column to X_max
        new_cpt = new_cpt.rename(columns={X: X + "_max"})

        return new_cpt
    
    
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
        
        new_cpt['p_x'] = new_cpt['p_x'] * new_cpt['p_y']

        # drop temporary columns
        new_cpt.drop(['p_y'], axis=1, inplace=True)
        new_cpt.rename(columns={'p_x': 'p'}, inplace=True)
        
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

    def ordering(self, x, strategy=None):
        '''
        Given a set of variables X in the Bayesian network, computes a good ordering for the elimination of X based on:
         - the min-degree heuristics (degree: how many edges the node has in the interaction graph)
         - the min-fill heuristics (order based on least amount of edges that have to be added to keep nodes connected in the interaction graph)

        input: x - set of variables
        strategy: 'min-degree' or 'min-fill'

        returns: list of ordering of those variables
        '''
        variables = set(x)

        # set strategy
        if strategy == 'min-degree':
            get_degree = self.get_degree
        elif strategy == 'min-fill':
            get_degree = self.get_fill_degree
        else:
            raise ValueError(f'Strategy {strategy} not implemented')

        interaction_graph = self.bn.get_interaction_graph()
        order = []

        while len(variables) > 0:
            # get variable with lowest degree
            degrees = {var: get_degree(var, interaction_graph) for var in variables}
            min_var = min(degrees, key=degrees.get)

            # add to order
            order.append(min_var)

            # prune interaction graph
            self.prune_interaction_graph(interaction_graph, min_var)
            
            # remove min_var from variables
            variables.remove(min_var)
        
        return order

    @staticmethod
    def get_degree(var: str, interaction_graph: Graph) -> int:
        """
        Returns the degree of the variable var in the interaction graph.
        :param var: The variable for which the degree should be returned.
        :return: The degree of the variable var in the interaction graph.
        """
        return len(interaction_graph.edges(var))
    
    def get_fill_degree(self, var: str, interaction_graph: Graph) -> int:
        """
        Returns the fill degree of the variable var in the interaction graph.
        :param var: The variable for which the fill degree should be returned.
        :return: The fill degree of the variable var in the interaction graph.
        """
        return self.prune_interaction_graph(interaction_graph.copy(), var)
        
    @staticmethod
    def prune_interaction_graph(interaction_graph: Graph, var: str) -> int:
        """
        Prunes the interaction graph by removing the node var and connecting all nodes connected to var together.

        Return: the number of edges that were added to the interaction graph
        """
        # connect all nodes connected to min_var together
        neighbors = [n for n in interaction_graph.neighbors(var)]
        added_edges = 0
        for i, node in enumerate(neighbors):
            for j in range(i+1, len(neighbors)):
                # if edge does not exist yet, add it
                if not interaction_graph.has_edge(node, neighbors[j]):
                    interaction_graph.add_edge(node, neighbors[j])
                    added_edges += 1
        
        # remove var node
        interaction_graph.remove_node(var)

        return added_edges
    
    def var_elimination(self, X, order_strategy='min-degree', max_out=False):
        '''
        given a set of variables X, eliminate variables in the right order
        optional input: ordering strategy: 'min-degree' or 'min-fill'. Standard: 'min-degree'
        returns: resulting factor
        '''
        if max_out:
            reduce = self.max_out
        else:
            reduce = self.sum_out
        # set cpts in multiplation order
        order = {}
        cpts = []
        for var in self.ordering(X, strategy=order_strategy):
            order[var] = []
            # get all factors that contain the variable
            for cpt_var, cpt in self.find_cpts_for_var(var).items():
                if cpt_var not in cpts:
                    cpts.append(cpt_var)
                    order[var].append(cpt)
        
        print(order.keys())
        # multiply factors in order
        factor = pd.DataFrame({'p': [1]})
        for var in order:
            # multiply all factors that contain the variable
            for cpt in order[var]:
                factor = self.multiply_factors(factor, cpt)
            # sum out variable
            factor = reduce(factor, var)
        
        # multiply all remaining factors
        left_overs = [var for var in self.bn.get_all_cpts() if var not in cpts]
        print(left_overs)
        for var in left_overs:
            factor = self.multiply_factors(factor, self.bn.get_cpt(var))

        return factor


    def marginal_distributions(self, query: List[str], evidence=pd.Series(), strategy='min-degree', posterior=True) -> pd.DataFrame:
        '''
        Given a query and evidence, return the marginal distribution of the query variables.
        Input:
            query: List of variable names.
            evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        Returns:
            marginal_distribution: A dictionary with the query variable names as keys and the corresponding marginal distributions as values.
        '''
        # set evidence
        self.set_evidence(evidence)

        # elimate variables not in query or evidence
        marginal = self.var_elimination([var for var in self.bn.get_all_variables() if var not in query], order_strategy=strategy)

        # if evidence, sum out q to compute posterior marginal
        if posterior:
            # normalize marginal
            print("Normalizing marginal")
            marginal['p'] = marginal['p'] / marginal['p'].sum()
        
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
        # set evidence
        self.set_evidence(evidence)
        
        # prune network
        self.network_pruning(self.bn.get_all_variables(), evidence)

        # eliminate all variables by maxing out
        mpe = self.var_elimination(self.bn.get_all_variables(), strategy, max_out=True)

        return mpe


    def map(self, query, evidence, strategy="min-fill"):
        """
        Given a query and some evidence, return the instantiation that maximizes the marginal distribition P(Q|e) along with the probability.
        
        Input:
            query: List of variable names to compute the marginal distribution for, given the evidence.
            evidence: Dictionary with variable names as keys and truth values as values.
            strategy: Strategy to order the variables by; "min-fill" or "min-degree".
        Returns:
            instantiation: A dictionary with all variable names as keys and the truth values for which the probability is maximized.
            probability: The probability of the instantiation.
        """
        # Compute joint marginal Pr(Q, e)
        marginal = self.marginal_distributions(query, evidence, strategy=strategy, posterior=False)

        print(marginal)
        
        # Max out all query variables to get the instantiation for which the probability is maximized
        map = marginal
        for var in query:
            map = self.max_out(map, var)
                    
        return map


if __name__ == '__main__':
    # Load the BN from the BIFXML file
    bnr = BNReasoner('testing\lecture_example.BIFXML')

    query = ["Winter?"]
    evidence = pd.Series({"Wet Grass?": False})

    print(bnr.map(query, evidence, strategy="min-fill"))

    bnr = BNReasoner('testing\lecture_example.BIFXML')
    print(bnr.mpe(evidence, strategy="min-fill"))



import BayesNet
from BNReasoner import BNReasoner
import random
import time

FILE = 'useable/win95pts.bifxml'

class Experiment:

    def __init__(self, file_path):
        """
        get reasoner object from file path
        """
        self.Reasoner = BNReasoner(file_path)

    # variable elimination with different orderings
    def ordering_strategy_experiment(self, percentage_vars, ordering_strategy):

        # get the variables in the BN (list of strings)
        variables = self.Reasoner.bn.get_all_variables()

        # get random sample of variables
        sample_size = int(len(variables) * percentage_vars)
        sample = random.sample(variables, sample_size)

        start = time.time()

        # eliminate the variables in the right order
        final_cpt = self.Reasoner.var_elimination(sample, strategy='min-fill')

        end = time.time()

        runtime = end - start

        return final_cpt, runtime

Experiment1 = Experiment(FILE)

result, runtime = (Experiment1.ordering_strategy_experiment(0.15, 'min_degree'))

print(f"this took {runtime*1000} ms")




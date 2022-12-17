import BayesNet
from BNReasoner import BNReasoner
import random
import time
from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd

FILES = ['asia', 'Berlin_clubs', 'cancer', 'dog_problem', 'earthquake', 'lecture_example', 'lecture_example2']

class Experiment:
    '''
    Runs into memory issues with the bigger networks (tables get too large)
    '''

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

    def marginal_distribution_experiment(self, query_size, evidence_size, ordering_strategy):

        # get the variables in the BN (list of strings)
        variables = self.Reasoner.bn.get_all_variables()

        # get query
        query = random.sample(variables, query_size)
        whats_left = [var for var in variables if var not in query]
        evidence = pd.Series({var: random.choice([True, False]) for var in random.sample(whats_left, evidence_size)})

        start = time.time()

        marginal = self.Reasoner.marginal_distributions(query, evidence, strategy=ordering_strategy)

        end = time.time()
        runtime = end - start

        return marginal, runtime


def experiment(percentage_vars, ordering_strategy, files, runs):

    results = []
    runtimes = []

    for file in tqdm(files):
        for run in range(runs):

            Experiment1 = Experiment(f'usable/{file}.bifxml')
            result, runtime = Experiment1.ordering_strategy_experiment(percentage_vars, ordering_strategy)
            results.append(result)
            runtimes.append(runtime)

    return results, runtimes




if __name__ == '__main__':

    # min_fill_resuts, min_fill_runtimes = experiment(0.8, 'min-fill', FILES, 1000)
    # min_degree_resuts, min_degree_runtimes = experiment(0.8, 'min-degree', FILES, 1000)
    # print(np.mean(min_fill_runtimes))
    # print(np.mean(min_degree_runtimes))

    # print(stats.ttest_ind(min_fill_runtimes, min_degree_runtimes))

    Experiment1 = Experiment('usable/win95pts.bifxml')
    # result, runtime = Experiment1.ordering_strategy_experiment(0.9, 'min-fill') 
    # print(f"This took {runtime*1000} milliseconds")

    marginal, runtime = Experiment1.marginal_distribution_experiment(5, 5, 'min-fill')
    print(runtime)




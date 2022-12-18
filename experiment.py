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


def experiment(ordering_strategy, files, runs):

    percentage_steps = np.arange(0.1, 1.0, 0.1)
    runtimes_per_percentage = []

    for i in percentage_steps:

        runtimes = []

        for file in tqdm(files):
            for run in range(runs):

                Experiment1 = Experiment(f'usable/{file}.bifxml')
                result, runtime = Experiment1.ordering_strategy_experiment(i, ordering_strategy)
                runtimes.append(runtime)
        
        runtimes_per_percentage.append(runtimes)

    return runtimes_per_percentage, percentage_steps



if __name__ == '__main__':

    min_fill_resuts, min_fill_runtimes = experiment('min-fill', FILES, 1000)
    min_degree_resuts, min_degree_runtimes = experiment('min-degree', FILES, 1000)
    print(np.mean(min_fill_runtimes, axis=1))
    print(np.mean(min_degree_runtimes, axis=1))

    # print(stats.ttest_ind(min_fill_runtimes, min_degree_runtimes))

    # result, runtime = Experiment1.ordering_strategy_experiment(0.9, 'min-fill') 
    # print(f"This took {runtime*1000} milliseconds")

   



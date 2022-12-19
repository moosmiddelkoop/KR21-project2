import BayesNet
from BNReasoner import BNReasoner
import random
import time
from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

FILES = ['asia', 'Berlin_clubs', 'cancer', 'dog_problem', 'earthquake', 'lecture_example', 'lecture_example2', 'lecture_example_alarm', 'lecture_example_simple']

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
        final_cpt = self.Reasoner.var_elimination(sample, order_strategy=ordering_strategy)

        end = time.time()

        runtime = (end - start) * 1000

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
    percentage_steps = [round(step, 1) for step in percentage_steps]
    runtimes_per_percentage = []

    for i in tqdm(percentage_steps):

        runtimes = []

        for file in files:
            for run in range(runs):

                Experiment1 = Experiment(f'usable/{file}.bifxml')
                result, runtime = Experiment1.ordering_strategy_experiment(i, ordering_strategy)
                runtimes.append(runtime)
        
        runtimes_per_percentage.append(runtimes)

    return runtimes_per_percentage, percentage_steps

def turn_results_into_dataframe(runtimes, percentages, ordering_strategy, dataframe=False, df=None):
    '''
    makes a dataframe of the runtiems per percentage and ordering strategy as hue
    '''

    if dataframe == False:

        df = pd.DataFrame()

        df['strategy'] = [ordering_strategy] * len(runtimes)
        df['percentage'] = percentages
        df['runtime'] = runtimes


        # row = {}
        # row['strat'] = ordering_strategy
        # for j, percentage in enumerate(percentages):
        #     row[str(percentage)] = runtimes[j]

        # df = df.append(row, ignore_index=True)

        # df['strat'] = ordering_strategy
        # for i, percentage in enumerate(percentages):
        #     df[str(percentage)] = runtimes[i]
        
    else:
        
        # df['strat'].append(pd.Series([ordering_strategy] * len(runtimes[0])))
        # for i in range(len(runtimes[0])):

        row = {}
        row['strategy'] = ordering_strategy
        for j, percentage in enumerate(percentages):
            row['percentage'] = percentage
            row['runtime'] = runtimes[j]

            df = df.append(row, ignore_index=True)

    return df


def plot_results(df):


    sns.lineplot(data=df, x='percentage', y='runtime', hue='strategy')
    plt.title('Mean runtime of variable elimination with different ordering strategies', pad=15)
    plt.xlabel('Fraction of variables eliminated')
    plt.ylabel('Runtime (ms)')
    plt.savefig('results1000.png')
    plt.show()

def regression(df, label):

    regres = stats.linregress(df['percentage'], df['runtime'])

    sns.regplot(data=df, x='percentage', y='runtime', label=label)
    plt.title('Regression of mean runtime of variable elimination \nwith different ordering strategies', pad=15)
    plt.xlabel('Fraction of variables eliminated')
    plt.ylabel('Runtime (ms)')
    plt.legend()

    return regres
    


if __name__ == '__main__':

    min_fill_runtimes, min_fill_percentages = experiment('min-fill', FILES, 1000)
    min_degree_runtimes, min_degree_percentages = experiment('min-degree', FILES, 1000)

    min_fill_df = turn_results_into_dataframe(np.mean(min_fill_runtimes, axis=1), min_fill_percentages, 'min-fill')
    min_deg_df = turn_results_into_dataframe(np.mean(min_degree_runtimes, axis=1), min_degree_percentages, 'min-degree')
    full_df = turn_results_into_dataframe(np.mean(min_degree_runtimes, axis=1), min_degree_percentages, 'min-degree', dataframe=True, df=min_fill_df)
    print(full_df)

    plot_results(full_df)
    regression_results = regression(min_fill_df, 'min-fill')
    regression_results2 = regression(min_deg_df, 'min-degree')

    print(regression_results)
    print(regression_results2)

    plt.savefig('regression1000.png')
    plt.show()


    # print(np.mean(min_fill_runtimes, axis= 1))
    # print(np.mean(min_degree_runtimes, axis= 1))

    

    # print(stats.ttest_ind(min_fill_runtimes, min_degree_runtimes))

    # result, runtime = Experiment1.ordering_strategy_experiment(0.9, 'min-fill') 
    # print(f"This took {runtime*1000} milliseconds")

   



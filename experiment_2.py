from BNReasoner import BNReasoner
import random
from copy import deepcopy
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

def experiment(bifxml_file, runs=100):
    """Run an over network experiment."""
    bnr_original = BNReasoner(bifxml_file)
    vars = bnr_original.bn.get_all_variables()
    
    results = []
    times = []
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        # generate random evidence
        evidence = pd.Series({var: random.choice([True, False]) for var in random.sample(vars, 2)})

        # mpe
        bnr = deepcopy(bnr_original)
        # time mpe
        start = time.time()
        mpe = bnr.mpe(evidence)
        end = time.time()
        mpe_time = end - start

        # map
        bnr = deepcopy(bnr_original)
        # time map
        start = time.time()
        map = bnr.map(vars, evidence)
        end = time.time()
        map_time = end - start

        times.append(map_time / mpe_time)


        # check how many values for map and mep agree
        correct = 0
        for var in mpe.columns:
            # don't compare p-values
            if var == 'p':
                continue

            if mpe[var].values[0] == map[var].values[0]:
                correct += 1


        results.append(correct / (len(mpe.columns) - 1))

    print(f"Fraction correct: {np.mean(results)}")
    print(f"Std: {np.std(results)}")

    return len(vars), np.mean(results), np.std(results), np.mean(times)


# loop over all files in the usable folder
files = os.listdir('usable')
results = {}
for file in tqdm(files):
    results[file] = experiment(f'usable/{file}')

nodes = []
avg = []
std = []
times = []
for file, result in results.items():
    print(f"{file}: {result}")
    nodes.append(result[0])
    avg.append(result[1])
    std.append(result[2])
    times.append(result[3])

# plot results
plt.errorbar(nodes, avg, yerr=std, fmt='o')
plt.show()
plt.errorbar(nodes, times, fmt='o')
plt.show()

import pandas as pd
from BNReasoner import BNReasoner

df = pd.read_csv('cpt.csv')
bnr = BNReasoner('testing\dog_problem.BIFXML')


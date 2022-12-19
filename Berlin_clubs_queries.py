from typing import List, Tuple, Dict, Union

from BayesNet import BayesNet
from BNReasoner import BNReasoner

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt



#%%

Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
all_cpts = Reasoner.bn.get_all_cpts()

all_cpts['Tourist'].to_latex()
all_cpts['Wasted'].to_latex()
all_cpts['DJ'].to_latex()
all_cpts['Guestlist'].to_latex()
all_cpts['Piercings'].to_latex()
all_cpts['Leather Outfit'].to_latex()
all_cpts['Queer Looks'].to_latex()
all_cpts['Enter at Night'].to_latex()
all_cpts['Enter Berghain'].to_latex()
all_cpts['Enter Matrix'].to_latex()







#%%


##########################
######## Berghain ########
##########################


# Prior Marginal
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Queer Looks', 'Enter Berghain', 'Tourist']
anti_query = [var for var in Reasoner.bn.get_all_variables() if var not in query]
print("Probability of a foreign / local entering Berghain and looking queer:")
result = Reasoner.var_elimination(anti_query, order_strategy = "min-fill")
result

#%%

# MPE
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Wasted": True, 'Enter Berghain': True})
print("The most likely scenario to get into Berghain as a wasted Tourist without being on the Guestlist:")
Reasoner.mpe(evidence, strategy = "min-fill")


#%%

#MAP
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['DJ', 'Enter Berghain']
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Enter at Night": False})
print("The most likely state of being a DJ and getting in, given that you're a Tourist trying to enter during the Day without being on the Guestlist")
Reasoner.map(query, evidence, strategy = "min-fill")


#%%

# Posterior Marginal 
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Wasted', 'Enter Berghain']
evidence = pd.Series({"Guestlist": False, "Queer Looks": True})
print("The probability of being wasted and getting into Berghain as a Queer looking person without being on the Guestlist:")
Reasoner.marginal_distributions(query, evidence, strategy = "min-fill")


#%%

######################
###### Matrix ########
######################


# Queries for our own network: Matrix

# Prior Marginal
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Enter at Night', 'Enter Matrix', 'Tourist']
anti_query = [var for var in Reasoner.bn.get_all_variables() if var not in query]
print("Probability of a foreign / local entering Matrix and looking queer:")
Reasoner.var_elimination(anti_query, order_strategy= "min-fill")


#%%

# MPE
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
evidence = pd.Series({"Tourist": False, "Wasted": False, 'Enter at Night': True, 'Enter Matrix': True})
print("The most likely scenario to get into Matrix as a wasted Tourist without being on the Guestlist:")
Reasoner.mpe(evidence, strategy = "min-fill")


#%%

#MAP
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['DJ', 'Guestlist']
evidence = pd.Series({"Tourist": False, "Wasted": True, "Enter Matrix" : True, "Enter at Night": True})
print("The most likely state of being a DJ and getting in, given that you're a wasted Tourist without being on the Guestlist")
Reasoner.map(query, evidence, strategy = "min-fill")


#%%

# Posterior Marginal 
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Wasted', 'Enter Matrix']
evidence = pd.Series({"Guestlist": False, "Leather Outfit": True, "Enter at Night": True})
print("The probability of being wasted and getting into Berghain as a Queer looking person without being on the Guestlist:")
Reasoner.marginal_distributions(query, evidence, strategy = "min-fill")


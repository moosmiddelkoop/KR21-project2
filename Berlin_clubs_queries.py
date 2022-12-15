from typing import List, Tuple, Dict, Union

from BayesNet import BayesNet
from BNReasoner import BNReasoner

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt


#%%

##########################
######## Berghain ########
##########################


# Prior Marginal
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Queer Looks', 'Enter Berghain', 'Tourist']
anti_query = [var for var in Reasoner.bn.get_all_variables() if var not in query]
print("Probability of a foreign / local DJ entering Berghain and looking queer:")
Reasoner.var_elimination(anti_query, strategy = "min-fill")

# Discoveries:
# Mostl likely: Not being a Tourist and not looking Queer and not getting in (plausible)
# With being a local with queer looks it is almost twice as likely to get into Berghain
# The same effect is apparent for Tourists, but not as strong
# Non anticipated: Non-queer looking Tourists are more likely to get in than non-queer looking locals

#%%

#MAP
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['DJ', 'Enter Berghain']
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Enter at Night": False})
print("The most likely state of being a DJ and getting in, given that you're a Tourist trying to enter during the Day without being on the Guestlist")
Reasoner.map(query, evidence, strategy = "min-fill")

# Answer: Being a DJ and Entering Berghain with a probability of 0.6394


#%%

# MPE
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Wasted": True, 'Enter Berghain': True})
print("The most likely scenario to get into Berghain as a wasted Tourist without being on the Guestlist:")
Reasoner.mpe(evidence, strategy = "min-fill")

# Answer: You're a DJ, don't try to enter at Night, and don't look queer (without Leather and Piercings). Probability: 0.0001454

#%%

# Posterior Marginal 
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Wasted', 'Enter Berghain']
evidence = pd.Series({"Guestlist": False, "Queer Looks": True})
print("The probability of being wasted and getting into Berghain as a Queer looking person without being on the Guestlist:")
Reasoner.marginal_distributions(query, evidence, strategy = "min-fill")


# Answer: Most likely not to be wasted and getting in, second to be wasted and not to get in. Queers: Don't get wasted!


#%%

######################
###### Matrix ########
######################


# Queries for our own network: Matrix

# Prior Marginal
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Queer Looks', 'Enter Matrix', 'Tourist']
anti_query = [var for var in Reasoner.bn.get_all_variables() if var not in query]
print("Probability of a foreign / local DJ being on the Guestlist:")
Reasoner.var_elimination(anti_query, strategy = "min-fill")

# Discoveries:
# Mostl likely: Not being a Tourist and not looking Queer and not getting in (plausible)
# With being a local with queer looks it is almost twice as likely to get into Berghain
# The same effect is apparent for Tourists, but not as strong
# Non anticipated: Non-queer looking Tourists are more likely to get in than non-queer looking locals

#%%

#MAP
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['DJ', 'Enter Matrix', 'Piercings']
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Wasted": True})
print("The most likely state of being a DJ and getting in, given that you're a wasted Tourist without being on the Guestlist")
Reasoner.map(query, evidence, strategy = "min-fill")

# Answer: Being a DJ and Entering Berghain with a probability of 0.6394


#%%

# MPE
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
evidence = pd.Series({"Tourist": True, "Guestlist": False, "Wasted": True, 'Enter Matrix': True})
print("The most likely scenario to get into Matrix as a wasted Tourist without being on the Guestlist:")
Reasoner.mpe(evidence, strategy = "min-fill")

# Answer: You're a DJ, don't try to enter at Night, and don't look queer (without Leather and Piercings). Probability: 0.0001454

#%%

# Posterior Marginal 
Reasoner = BNReasoner('testing/Berlin_clubs.BIFXML')
query = ['Wasted', 'Enter Matrix']
evidence = pd.Series({"Guestlist": False, "Queer Looks": True})
print("The probability of being wasted and getting into Berghain as a Queer looking person without being on the Guestlist:")
Reasoner.marginal_distributions(query, evidence, strategy = "min-fill")


# Answer: Most likely not to be wasted and getting in, second to be wasted and not to get in. Queers: Don't get wasted!

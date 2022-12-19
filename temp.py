import pandas as pd
from BNReasoner import BNReasoner
import os

bn_info = pd.DataFrame(columns=['file', 'nodes', 'edges'])
# for every file in usable
for file in os.listdir('usable'):
    # add info to dataframe
    bnr = BNReasoner(f'usable/{file}')
    bn_info = bn_info.append({'file': file, 'nodes': len(bnr.bn.get_all_variables()), 'edges': len(bnr.bn.structure.edges)}, ignore_index=True)

# save to latex table
print(bn_info.to_latex(index=False))

from BNReasoner import BNReasoner
from copy import deepcopy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# load BNReasoner objects
dog_problem_bnr = BNReasoner('testing\dog_problem.BIFXML')
example_1_bnr =  BNReasoner('testing\lecture_example.BIFXML')
example_2_bnr =  BNReasoner('testing\lecture_example2.BIFXML')
example_alarm = BNReasoner('testing\lecture_example_alarm.BIFXML')
example_simple = BNReasoner('testing\lecture_example_simple.BIFXML')

bnrs = [dog_problem_bnr, example_1_bnr, example_2_bnr, example_alarm, example_simple]

# show graphs
for bnr in bnrs:
    bnr.bn.draw_structure()

# print variables for each BNReasoner
for bnr in bnrs:
    print(bnr.bn.get_all_variables())

# tests for find_leave_nodes
assert(set(dog_problem_bnr.find_leaf_nodes()) == set(['hear-bark', 'light-on']))
assert(set(example_1_bnr.find_leaf_nodes()) == set(['Wet Grass?', 'Slippery Road?']))
assert(set(example_2_bnr.find_leaf_nodes()) == set(['O']))

# tests for is_connected

# since all nodes should be connected to each other, all should return True
for bnr in bnrs:
    vars = bnr.bn.get_all_variables()
    for i, start in enumerate(vars):
        for end in vars[i+1:]:
            assert(bnr.is_connected(start, end))

# test for is_connected with disconnected nodes
temp = deepcopy(example_2_bnr)
temp.bn.del_edge(("I", "X"))
assert(not temp.is_connected("I", "X"))

# test for connection to self
assert(example_2_bnr.is_connected("I", "I"))

# tests for find_cpts_per_var
for bnr in bnrs:
    all_vars = bnr.bn.get_all_variables()
    cpts = {var: bnr.find_cpts_for_var(var) for var in all_vars}
    for var in cpts:
        assert(var in all_vars)
        assert(var in cpts[var])

        for var_2 in cpts[var]:
            assert(var in bnr.bn.get_cpt(var_2).columns)

# tests for set_evidence
for bnr in bnrs:
    vars = bnr.bn.get_all_variables()
    evidence = {}
    # set evidence for all variables to True
    # this is done expandingly to test several evidence lenghts
    for var in vars:
        bnr = deepcopy(bnr)
        evidence[var] = True
        bnr.set_evidence(pd.Series(evidence))

    # check that no cpt has False in it (exclude p column to avoid 0.0 probability recognised as False)
    assert(all([False not in cpt.loc[ : , cpt.columns != 'p'].to_numpy().flatten() for cpt in bnr.bn.get_all_cpts().values()]))

# test pruning
bnr = deepcopy(example_1_bnr)
evidence = pd.Series({"Winter?": True, "Rain?": False})
query = ["Wet Grass?"]

bnr.network_pruning(query, evidence)

# check if winter and rain are completely disconnected
interaction_graph = bnr.bn.get_interaction_graph()
assert(len(interaction_graph.edges("Winter?")) == 0)
assert(len(interaction_graph.edges("Rain?")) == 0)

# check if sprinkler and wet grass are connected
assert(bnr.bn.get_children("Sprinkler?") == ["Wet Grass?"])

# check if cpts are correct, see lecture notes for correct values: lecture 3, slide 31
[print(f"{var}\n{cpt}") for var, cpt in bnr.bn.get_all_cpts().items()]

# test for d-separation
# test lecture examples: lecture 2 slide 31
bnr = deepcopy(example_alarm)
assert(bnr.is_d_seperated({"R"}, {"B"}, {"E", "C"}))

bnr = deepcopy(example_alarm)
assert(bnr.is_d_seperated({"E"}, {"B"}, {}))

bnr = deepcopy(example_alarm)
assert(not bnr.is_d_seperated({"E"}, {"B"}, {"C"}))

bnr = deepcopy(example_alarm)
assert(not bnr.is_d_seperated({"R"}, {"C"}, {}))

# tests for summing out, lecture 3, slide 8
bnr = deepcopy(example_1_bnr)
summed_out = bnr.sum_out(bnr.bn.get_cpt("Wet Grass?"), "Wet Grass?")
summed_out = bnr.sum_out(summed_out, "Rain?")
summed_out = bnr.sum_out(summed_out, "Sprinkler?")
assert(summed_out['p'].tolist() == [4.0])

# tests for maxing out
bnr = deepcopy(example_1_bnr)
maxed_out = bnr.max_out(bnr.bn.get_cpt("Wet Grass?"), "Wet Grass?")
print(maxed_out)

# tests for ordering
for bnr in bnrs:
    bnr = deepcopy(bnr)
    # show interaction graph for visual inspection
    nx.draw(bnr.bn.get_interaction_graph(), with_labels=True)
    # min degree ordering
    print("min degree ordering: ", bnr.ordering(bnr.bn.get_all_variables(), "min-degree"))
    # min fill ordering
    print("min fill ordering: ", bnr.ordering(bnr.bn.get_all_variables(), "min-fill"))
    plt.show()

# test var elimination
bnr = deepcopy(example_1_bnr)
bnr.var_elimination(["Wet Grass?", "Sprinkler?", "Rain?"])

bnr = deepcopy(example_simple)
[print(f"{var}\n{cpt}") for var, cpt in bnr.bn.get_all_cpts().items()]
bnr.var_elimination(["B", "A"])
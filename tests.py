from BNReasoner import BNReasoner
from copy import deepcopy

# load BNReasoner objects
dog_problem_bnr = BNReasoner('testing\dog_problem.BIFXML')
example_1_bnr =  BNReasoner('testing\lecture_example.BIFXML')
example_2_bnr =  BNReasoner('testing\lecture_example2.BIFXML')

bnrs = [dog_problem_bnr, example_1_bnr, example_2_bnr]

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
    cpts = bnr.find_cpts_per_var()
    all_vars = bnr.bn.get_all_variables()
    for var in cpts:
        assert(var in all_vars)
        assert(var in cpts[var])

        for var_2 in cpts[var]:
            assert(var in bnr.bn.get_cpt(var_2).columns)

# tests for set_evidence
for bnr in bnrs:
    vars = bnr.bn.get_all_variables()
    for var in vars:
        bnr.set_evidence(var, True)
        assert(bnr.bn.get_evidence(var) == True)
        bnr.set_evidence(var, False)
        assert(bnr.bn.get_evidence(var) == False)


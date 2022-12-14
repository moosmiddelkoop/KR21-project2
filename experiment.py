import BayesNet
import BNReasoner

FILE = 'bifxml/small/asia.bifxml'

class Experiment(file_path):

    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # variable elimination with different orderings
    def ordering_strategy_experiment(BN):
        pass


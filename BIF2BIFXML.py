'''
convert a .bif file into a .bifxml file
accepts two command line arguments: a relative input path and a relative output path
'''

import pyAgrum
import sys

# input_path = sys.argv[1]
# output_path = sys.argv[2]

INPUT_PATH = "bif/small/survey.bif"
OUTPUT_PATH = "bifxml/small/survey.bifxml"

# load the bif file
bn = pyAgrum.loadBN(f"{INPUT_PATH}")

# save as bifxml
bn.saveBIFXML(f"{OUTPUT_PATH}")





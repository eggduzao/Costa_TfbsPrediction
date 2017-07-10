#################################################################################################
# Counts the number of base pairs that a bed file is spanning
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. separator = The separator in the bed file. Can be \"tab\" or \"spc\".")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("    3. [inputList] = List of .bed files (one for each replicate).")
params.append("###")
params.append("Output: ")
params.append("    1. [<chrName>.bed] = List of files per chrom. with replicates merged.")
params.append("###")
#################################################################################################

# Import
import os
import sys
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
bedFileName = sys.argv[1]

# Execution
bedFile = open(bedFileName,"r")
total = 0
for line in bedFile:
    ll = line.strip().split("\t")
    total += (int(ll[2])-int(ll[1]))
bedFile.close()

print total



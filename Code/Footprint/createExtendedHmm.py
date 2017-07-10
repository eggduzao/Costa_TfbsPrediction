#################################################################################################
# Adds two transitions to the default HMM model.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. hmmFileName = File containing MBPSs.")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <hmmFileName>_ext.hmm = Result evidence bed file.")
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
hmmFileName = sys.argv[1]
outputLocation = sys.argv[2]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Reading hmm
hmmStates, dimNo, pi, A, E = hmmFunctions.createHMM(hmmFileName,returnMode="mat")

# Updating transitions
totalTrans1 = A[0][1]
A[0][1] = (1.0*totalTrans1) / 3.0 # Update old transition 
A[0][4] = (2.0*totalTrans1) / 3.0 # New transition
totalTrans2 = A[6][1]
A[6][1] = (1.0*totalTrans2) / 3.0 # Update old transition 
A[6][0] = (2.0*totalTrans2) / 3.0 # New transition

# Writing new HMM
outputFileName = hmmFileName.split("/")[-1].split(".")[0]+"_ext.hmm"
hmmFunctions.writeHMM(pi, A, E, outputLocation+outputFileName, precision=5)



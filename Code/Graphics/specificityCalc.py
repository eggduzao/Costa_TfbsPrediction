#################################################################################################
# Creates a ROC curve from a bed file already separated by evidence. True positives are marked
# with Y in the NAME field and false positives are marked with N.
# This differs from the regular rocFromBed because it takes into account the fact that the method
# is site-centric (SC) or segmentation-based (SB).
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the MPBS. The same name as in the NAME field.")
params.append("    2. typeList = List of file types (comma separated): SB or SC")
params.append("    3. labelList = List of labels to each of the bed files in the list.")
params.append("    4. bedList = List of files containing bed coordinates separated by comma.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <mpbsName>_fpr.txt = ROC graph.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
import operator
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsName = sys.argv[1]
typeList = sys.argv[2].split(",")
labelList = sys.argv[3].split(",")
bedList = sys.argv[4].split(",")
outputLocation = sys.argv[5]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Standardize function
def standardize(vec):
    maxN = max(vec)
    minN = min(vec)
    return [(e-minN)/(maxN-minN) for e in vec]

# Numerology function
def numerology(n):
    ret = sum([ord(e) for e in n])
    while(len(str(ret)) > 1):
        ret = sum([int(e) for e in str(ret)])
    return ret

# Parameters
maxPoints = 10000
fpr_auc = 0.1
cellType = outputLocation.split("/")[-2]

#############################################################
# Calculating ROC
#############################################################

# Reading data points

for k in range(0,len(bedList)):

    # Initialization
    bedFileName = bedList[k]
    




#############################################################
# Writing Results
#############################################################



























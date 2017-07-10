#################################################################################################
# Creates a heatmap based on a sequence file. Each color corresponds to a nucleotide.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. fastaFile = If y then the '.' is also considered a negative instance.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <mpbsName>.<outExt> = ROC graph.")
params.append("    2. <mpbsName>_sens.txt = Average % sensitivity at 1% FDR.")
params.append("    3. <mpbsName>_auc.txt = AUC in txt format.")
params.append("    4. <mpbsName>_roc.txt = ROC in txt format.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
import operator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import auc
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
allNeg = sys.argv[1]
outExt = sys.argv[2]
mpbsName = sys.argv[3]
labelList = sys.argv[4].split(",")
bedList = sys.argv[5].split(",")
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Parameters
maxPoints = 10000
fpr_auc = 0.1

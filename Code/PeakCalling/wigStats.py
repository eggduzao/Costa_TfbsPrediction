#################################################################################################
# Evaluates the mean, std and var for each chromosome and for the whole genome
# over genome-wide wig signals.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. inputLocation = Input location of the genomic signal.")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. stats.txt = Statistics of the genomic signal.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
inputLocation = sys.argv[1]
if(inputLocation[-1]!="/"): inputLocation += "/"
outputLocation = sys.argv[2]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Iterating on chrList
chrList = ["chr"+str(e) for e in range(1,23)] + ["chrX"]
outFile = open(outputLocation+"stats.txt","w")
outFile.write("SIGNAL MEAN VAR STD VAR-1 STD-1\n")
globalSum = 0.0
globalSqSum = 0.0
globalTotal = 0.0
for chrName in chrList:

    # Local Structures
    chrSum = 0.0
    chrSqSum = 0.0
    chrTotal = 0.0

    # Calculating descriptors
    if(not os.path.exists(inputLocation+chrName+".wig")): continue
    inFile = open(inputLocation+chrName+".wig","r")
    for line in inFile:
        if(line[0] == "f"): continue
        v = float(line.strip())
        globalSum += v; chrSum += v
        globalSqSum += (v**2.0); chrSqSum += (v**2.0)
        globalTotal += 1.0; chrTotal += 1.0

    # Evaluating mean, var and std
    mean = chrSum / chrTotal
    var = (chrSqSum-((chrSum**2)/chrTotal)) / (chrTotal); std = math.sqrt(var)
    var_1 = (chrSqSum-((chrSum**2)/chrTotal)) / (chrTotal-1); std_1 = math.sqrt(var_1)
    outFile.write(chrName+" "+str(mean)+" "+str(var)+" "+str(std)+" "+str(var_1)+" "+str(std_1)+"\n")

    # Termination
    inFile.close()

# Global mean, var and std
mean = globalSum / globalTotal
var = (globalSqSum-((globalSum**2)/globalTotal)) / (globalTotal); std = math.sqrt(var)
var_1 = (globalSqSum-((globalSum**2)/globalTotal)) / (globalTotal-1); std_1 = math.sqrt(var_1)
outFile.write("genome "+str(mean)+" "+str(var)+" "+str(std)+" "+str(var_1)+" "+str(std_1)+"\n")

# Termination
outFile.close()



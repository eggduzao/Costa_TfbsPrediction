#################################################################################################
# Call peaks based on a p-value by fitting data to a gamma distribution
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. pValue = p-value to stablish the cutoff.")
params.append("    2. statsFileName = Location and name of the stats file.")
params.append("    3. inputLocation = Input location of the genomic signal.")
params.append("    4. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. peaks_<pValue>.bed = Called peaks.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
from scipy.stats import gamma
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
pValue = float(sys.argv[1])
statsFileName = sys.argv[2]
inputLocation = sys.argv[3]
if(inputLocation[-1]!="/"): inputLocation += "/"
outputLocation = sys.argv[4]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Fetching mean and variance
mean = 0.0; var = 0.0
# TODO

# Gamma distribution
beta = var / mean
alpha = mean / beta
cutoff = gamma.ppf((1.0-pValue),alpha,scale=beta)
maxV = gamma.ppf((0.9999),alpha,scale=beta)

# Calling peaks
outFile = open(outputLocation+"peaks_"+str(pValue)+".bed","w")
chrList = ["chr"+str(e) for e in range(1,23)] + ["chrX"]
counter = 1
for chrName in chrList:
    
    # Iterating on input File
    if(not os.path.exists(inputLocation+chrName+".wig")): continue
    inFile = open(inputLocation+chrName+".wig","r")
    currPos = 0; flagStart = False; peakStart = 0; summitValue = -1.0
    for line in inFile:
        if(line[0] == "f"):
            currPos = int(line.strip().split(" ")[2][6:])
            continue
        v = float(line.strip())
        if(flagStart):
            if(v > summitValue):
                summitValue = v              
            if(v < cutoff):
                outFile.write(chrName+" "+str(peakStart-1)+" "+str(currPos-1)+" p"+str(counter)+" "+str(min(int(((summitValue-cutoff)*1000)/(maxV-cutoff)),1000))+" .\n")
                flagStart = False
                counter += 1
        else:
            if(v >= cutoff):
                peakStart = currPos
                flagStart = True
                summitValue = v
        currPos += 1

    # Termination
    inFile.close()

# Termination
outFile.close()



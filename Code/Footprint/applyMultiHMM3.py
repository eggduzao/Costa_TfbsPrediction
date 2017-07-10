#################################################################################################
# Apply the multivariate HMM to the entire genome for n>1 signals.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. coordFileName = Coordinates to apply the HMM.")
params.append("    2. hmmFileName = The location + name of the proximal hmm file.")
params.append("    3. signalList = The locations of the signals files.")
params.append("    4. outputFileName = Output file name.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Resulting footprints as a color bed.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
import math
import operator
import numpy as np
from ghmm import *
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
coordFileName = sys.argv[1]
hmmFileName = sys.argv[2]
signalList = sys.argv[3].split(",")
outputFileName = sys.argv[4]

##################################################
### Parameters
##################################################

# HMM
usePosterior = True

# Footprint color bed output
stateNameDict = dict([(0,"BACK"), (1,"UPH"), (2,"TOPH"), (3,"DOWNH"), (4,"UPD"), (5,"TOPD"), (6,"DOWND"), (7,"FP")])
colorDict = dict([(0,"50,50,50"), (1,"110,250,110"), (2,"90,180,240"), (3,"255,80,90"), (4,"10,80,0"), (5,"20,40,150"), (6,"150,20,40"), (7,"198,150,0")])

##################################################
### Applying HMM and creating posteriorList
##################################################

# Creating hmms
hmm, hmmStates, dimNo, emissionDomain = hmmFunctions.createHMM(hmmFileName,returnMode="hmm")

# Opening signals
signalFileList = []; signalBwList = []
for signalFileName in signalList:
    signalFileList.append(open(signalFileName,"r"))
    signalBwList.append(BigWigFile(signalFileList[-1]))

# Create output files list
coordName = coordFileName.split("/")[-1].split(".")[0]
outputBedFile = open(outputFileName,"w")

# Iterating on coordinate file
coordFile = open(coordFileName,"r")
for line in coordFile:

    # Fetching coordinate
    ll = line.strip().split("\t")
    chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])

    # Fetching signal sequences
    try:
        seqVect = aux.vectorizeByCol([aux.correctBW(bw.get(chrName,p1,p2),p1,p2) for bw in signalBwList])
    except Exception:
        continue

    if(usePosterior):

        # Evaluating posterior vector
        try:
            emissionSeq = EmissionSequence(emissionDomain,seqVect)
            posteriorList = hmm.posterior(emissionSeq)[:(len(emissionSeq)/dimNo)]
        except NameError: continue

        # Writing bed file
        currStart = p1
        currPos = currStart + 1
        currV = posteriorList[0].index(max(posteriorList[0]))
        for v in [e.index(max(e)) for e in posteriorList[1:]]:
            if(v != currV):
                outputBedFile.write(chrName+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")
                currStart = currPos
                currV = v
            currPos += 1
        outputBedFile.write(chrName+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")

    else:

        # Evaluate posterior vector
        try:
            emissionSeq = EmissionSequence(emissionDomain,seqVect)
            posteriorList = hmm.viterbi(emissionSeq)[0]
        except Exception: continue
        
        # Writing bed file
        currStart = p1
        currPos = currStart + 1
        try:
            currV = posteriorList[0]
        except Exception:
            continue
        for v in posteriorList[1:]:
            if(v != currV):
                outputBedFile.write(chrName+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")
                currStart = currPos
                currV = v
            currPos += 1
        outputBedFile.write(chrName+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")

# Termination
coordFile.close()
outputBedFile.close()
for e in signalFileList: e.close()



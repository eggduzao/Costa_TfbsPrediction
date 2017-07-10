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
from sklearn import hmm
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
covarType = "full"

# Footprint color bed output
stateNameDict = dict([(0,"BACK"), (1,"UPD"), (2,"TOPD"), (3,"DOWND"), (4,"FP")])
colorDict = dict([(0,"50,50,50"), (1,"10,80,0"), (2,"20,40,150"), (3,"150,20,40"), (4,"198,150,0")])

##################################################
### Applying HMM and creating posteriorList
##################################################

# Creating hmm
hmmStates, dimNo, startprob, transmat, means, covars = hmmFunctions.createHMM(hmmFileName,returnMode="sci")
hmm = hmm.GaussianHMM(n_components=hmmStates, covariance_type=covarType, transmat=np.array(transmat), startprob=np.array(startprob))
hmm.means_ = np.array(means); hmm.covars_ = np.array(covars)

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
    
    # Fetching signal
    try:
        seqVect = np.array([aux.correctBW(bw.get(chrName,p1,p2),p1,p2) for bw in signalBwList])
        seqVect = seqVect.T
    except Exception:
        continue
    
    # Applying HMM
    try:
        posteriorList = hmm.predict(seqVect)
    except Exception:
        continue

    # Writing bed file
    currStart = p1
    currPos = currStart + 1
    currV = posteriorList[0]
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



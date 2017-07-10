
# Import
import os
import sys
import glob
from Bio import Motif

# Input
inLoc = "/home/egg/hpcwork/izkf/projects/egg/Data/PWM/Jaspar_Transfac/"
inList = glob.glob(inLoc+"*.pfm")
inList.sort()
motifList = [ e.split("/")[-1].split(".")[0] for e in inList]
outFileName = "/home/egg/hpcwork/izkf/projects/egg/Data/PWM/ic.txt"
outFile = open(outFileName,"w")

# Execution
for motif in motifList:
    pwmFile = open(inLoc+motif+".pfm","r")
    pwm = Motif.read(pwmFile,"jaspar-pfm")
    outFile.write("\t".join([motif,str(round(pwm.ic(),2))])+"\n")
    pwmFile.close()

# Termination
outFile.close()



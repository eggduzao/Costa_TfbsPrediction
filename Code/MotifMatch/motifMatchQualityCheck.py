#################################################################################################
# Based on a bed with MPBSs, fetch the sequences and evaluates statistics to access the
# quality of the motif matching procedure.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. specIntList = List of intervals of specific bases.")
params.append("    2. pwmFileName = Pwm file name in jaspar format.")
params.append("    3. bedFileName = Bed file name.")
params.append("    4. fastaFileName = File containing the PWM in the JASPAR format.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <bedFileName>_stats<Spec>.txt = Statistics on number of mismatches.")
params.append("    2. <bedFileName>_distMis<Spec>.png = Mismatches distribution.")
params.append("    3. <bedFileName>_distPos<Spec>.png = Mismatches distribution by motif position.")
params.append("    4. <bedFileName>_scatter<Spec>.png = Scatterplot of MM score vs. # mismatches.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import numpy as np
from Bio import Motif
import matplotlib as mpl
mpl.use('Agg') # As the server does not have the $DISPLAY env variable, this fix had to be added.
import matplotlib.pyplot as plt
import pylab
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
specIntList = [[int(k) for k in e.split(",")] for e in sys.argv[1].split(":")]
pwmFileName = sys.argv[2]
bedFileName = sys.argv[3]
fastaFileName = sys.argv[4]
outputLocation = sys.argv[5]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Revert function
revDict=dict([("A","T"),("T","A"),("C","G"),("G","C"),("N","N")])
def reverse(sequence):
    retSeq = []
    for c in sequence[::-1]: retSeq.append(revDict[c])
    return "".join(retSeq)

# Fetching sequences
bedName = bedFileName.split("/")[-1][:-4]
os.system("fastaFromBed -fi "+fastaFileName+" -fo "+outputLocation+bedName+".txt"+" -bed "+bedFileName+" -tab")

# Reading pwm
pwmFile = open(pwmFileName,"r")
pwm = Motif.read(pwmFile,"jaspar-pfm")
motif = str(pwm.consensus()).upper()
pwmFile.close()

# Reading input vectors
misVec = []; misVecSpec = []
posVec = []; posVecSpec = []
scoreVec = [];
bedFile = open(bedFileName,"r")
seqFile = open(outputLocation+bedName+".txt","r")
for bedLine in bedFile:

    # Reading line
    seqLine = seqFile.readline()
    bedL = bedLine.strip().split("\t")
    seqL = seqLine.strip().split("\t")
    scoreVec.append(int(bedL[4]))
    seq = seqL[1].upper()
    if(bedL[5] == "-"): seq = reverse(seqL[1].upper())

    # Whole sequence stats
    misN = 0
    for i in range(0,len(motif)):
        if(seq[i] != motif[i]):
            misN += 1
            posVec.append(i)
    misVec.append(int(misN))

    # Specific sequence stats
    misN = 0
    for v in specIntList:
        for i in range(v[0],v[1]):
            if(seq[i] != motif[i]):
                misN += 1
                posVecSpec.append(i)
    misVecSpec.append(int(misN))
bedFile.close()
seqFile.close()
os.system("rm "+outputLocation+bedName+".txt")

# Write statistics - normal
statsFile = open(outputLocation+bedName+"_stats.txt","w")
statsFile.write("Total # Mismatches:\t"+str(sum(misVec))+"\n")
statsFile.write("Total # Sequences:\t"+str(len(misVec))+"\n")
statsFile.write("Total Motif Length:\t"+str(len(motif))+"\n")
misVec = np.array(misVec)
statsFile.write("Mean:\t"+str(misVec.mean())+"\n")
statsFile.write("Var:\t"+str(misVec.var())+"\n")
statsFile.write("Std:\t"+str(misVec.std())+"\n")
statsFile.close()

# Write statistics - spec
statsFile = open(outputLocation+bedName+"_statsSpec.txt","w")
statsFile.write("Total # Mismatches:\t"+str(sum(misVecSpec))+"\n")
statsFile.write("Total # Sequences:\t"+str(len(misVecSpec))+"\n")
specSize = 0
for v in specIntList: specSize += (v[1]-v[0])
specVec = []
for v in specIntList:
    for e in range(v[0],v[1]): specVec.append(e)
statsFile.write("Total Motif Length:\t"+str(specSize)+"\n")
misVecSpec = np.array(misVecSpec)
statsFile.write("Mean:\t"+str(misVecSpec.mean())+"\n")
statsFile.write("Var:\t"+str(misVecSpec.var())+"\n")
statsFile.write("Std:\t"+str(misVecSpec.std())+"\n")
statsFile.close()

# Mismatch distribution graph - normal
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.hist(misVec, bins=range(0,len(motif)+2), facecolor="black")
ax.set_title("Mismatch Distribution - All")
ax.set_xlabel("# Mismatches")
ax.set_xticks(np.arange(0.5,len(motif)+1,1.0))
ax.set_xticklabels(np.arange(0,len(motif)+1,1))
pylab.xlim([-0.5,len(motif)+1.5])
fig.savefig(outputLocation+bedName+"_distMis.png", format="png", dpi=300, bbox_inches='tight')

# Mismatch distribution graph - spec
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.hist(misVecSpec, bins=range(0,specSize+2), facecolor="black")
ax.set_title("Mismatch Distribution - Spec")
ax.set_xlabel("# Mismatches")
ax.set_xticks(np.arange(0.5,specSize+1,1.0))
ax.set_xticklabels(np.arange(0,specSize+1,1))
pylab.xlim([-0.5,specSize+1.5])
fig.savefig(outputLocation+bedName+"_distMisSpec.png", format="png", dpi=300, bbox_inches='tight')

# Position distribution graph - normal
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.hist(posVec, bins=range(0,len(motif)+1), facecolor="black")
ax.set_title("Mismatch Position Distribution - All")
ax.set_xlabel("Motif Positions")
ax.set_ylabel("# Mismatches")
ax.set_xticks(np.arange(0.5,len(motif),1.0))
ax.set_xticklabels(np.arange(0,len(motif),1))
pylab.xlim([-0.5,len(motif)+0.5])
fig.savefig(outputLocation+bedName+"_distPos.png", format="png", dpi=300, bbox_inches='tight')

# Position distribution graph - spec
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.hist(posVecSpec, bins=specVec+[specVec[-1]+1], facecolor="black")
ax.set_title("Mismatch Position Distribution - Spec")
ax.set_xlabel("Motif Positions")
ax.set_ylabel("# Mismatches")
ax.set_xticks([e+0.5 for e in specVec])
ax.set_xticklabels(specVec+[specVec[-1]+1])
pylab.xlim([specVec[0]-0.5,specVec[-1]+1.5])
fig.savefig(outputLocation+bedName+"_distPosSpec.png", format="png", dpi=300, bbox_inches='tight')

# Scatterplot - normal
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.grid(True, which='both')
ax.plot(misVec, scoreVec, 'o', alpha = 0.5, color="black")
ax.set_title("MM Score vs. # Mismatches - All")
ax.set_xlabel("# Mismatches")
ax.set_ylabel("MM Score")
ax.set_xticks(np.arange(0,len(motif)+1,1))
ax.set_xticklabels(np.arange(0,len(motif)+1,1))
pylab.xlim([-0.5,len(motif)+0.5])
fig.savefig(outputLocation+bedName+"_scatter.png", format="png", dpi=300, bbox_inches='tight')

# Scatterplot - spec
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.grid(True, which='both')
ax.plot(misVecSpec, scoreVec, 'o', alpha = 0.5, color="black")
ax.set_title("MM Score vs. # Mismatches - Spec")
ax.set_xlabel("# Mismatches")
ax.set_ylabel("MM Score")
ax.set_xticks(np.arange(0,specSize+1,1))
ax.set_xticklabels(np.arange(0,specSize+1,1))
pylab.xlim([-0.5,specSize+0.5])
fig.savefig(outputLocation+bedName+"_scatterSpec.png", format="png", dpi=300, bbox_inches='tight')



#################################################################################################
# Creates signals to be used by CENTIPEDE software.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the motif.")
params.append("    2. mpbsFileName = MPBS file name separated by evidence.")
params.append("    3. consFileName = Conservation file name.")
params.append("    4. tssFileName = TSS distance file name.")
params.append("    5. dnaseTotExt = Total window size length for DNase.")
params.append("    6. dnasePosFileName = File containing DNase counts for positive strand.")
params.append("    7. dnaseNegFileName = File containing DNase counts for negative strand.")
params.append("    8. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <mpbsName>_PRIOR.txt = Priors in tab separated format.")
params.append("    2. <mpbsName>_DNASE.bed = DNase counts in tab separated format.")
params.append("###")
#################################################################################################

# Import
import os
import sys
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsName = sys.argv[1]
mpbsFileName = sys.argv[2]
consFileName = sys.argv[3]
tssFileName = sys.argv[4]
dnaseTotExt = int(sys.argv[5])
dnasePosFileName = sys.argv[6]
dnaseNegFileName = sys.argv[7]
outputLocation = sys.argv[8]
if(outputLocation[-1] != "/"): outputLocation+="/"

#############################################################################################
### MPBS
#############################################################################################

# Reading mpbs list
mpbsFile = open(mpbsFileName,"r")
mpbsList = []
for line in mpbsFile:
    ll = line.strip().split("\t")
    if(ll[3].split(":")[0] == mpbsName): mpbsList.append(ll)
mpbsFile.close()

#############################################################################################
### Prior
#############################################################################################

# Initializing prior matrix with MPBS scores
priorMatrix = []
for coord in mpbsList: priorMatrix.append([float(coord[4])])

# Creating conservation prior
consFile = open(consFileName,"r")
bw = BigWigFile(consFile)
for i in range(0,len(mpbsList)):
    coord = mpbsList[i]
    chrName = coord[0]; pos1 = int(coord[1]); pos2 = int(coord[2])
    bwQuery = aux.correctBW(bw.get(chrName,pos1,pos2),pos1,pos2)
    priorMatrix[i].append(sum(bwQuery)/float(len(bwQuery)))
consFile.close()

# Creating TSS distance prior
tssFile = open(tssFileName,"r")
bw = BigWigFile(tssFile)
for i in range(0,len(mpbsList)):
    coord = mpbsList[i]
    chrName = coord[0]; pos1 = int(coord[1]); pos2 = int(coord[2])
    mid = (pos1+pos2)/2
    bwQuery = aux.correctBW(bw.get(chrName,mid,mid+1),mid,mid+1)
    priorMatrix[i].append(bwQuery[0])
tssFile.close()

# Writing prior
outputPriorFileName = outputLocation+mpbsName+"_PRIOR.txt"
outputPriorFile = open(outputPriorFileName,"w")
for e in priorMatrix: outputPriorFile.write("\t".join([str(k) for k in e])+"\n")
outputPriorFile.close()

#############################################################################################
### DNase
#############################################################################################

# DNase input files
dnasePosFile = open(dnasePosFileName,"r"); bwPos = BigWigFile(dnasePosFile)
dnaseNegFile = open(dnaseNegFileName,"r"); bwNeg = BigWigFile(dnaseNegFile)

# DNase output footprint
dnaseTempFileName = outputLocation+mpbsName+"_DNASE.txt"
dnaseTempFile = open(dnaseTempFileName,"w")

# Iterating on coordinates to fetch the signals
for coord in mpbsList:

    # Getting coordinate info
    chrName = coord[0]
    pos1 = int(coord[1])-(dnaseTotExt/2)
    pos2 = int(coord[2])+(dnaseTotExt/2)
    strand = coord[5]

    # Fetching sequence
    bwQueryPos = aux.correctBW(bwPos.get(chrName,max(pos1,0),pos2),pos1,pos2)
    bwQueryNeg = aux.correctBW(bwNeg.get(chrName,max(pos1,0),pos2),pos1,pos2)

    # Writing footprint signal // Limiting counts to 30bp
    dnaseTempFile.write("\t".join([str(min(e,30.0)) for e in bwQueryPos])+"\t"+"\t".join([str(min(e,30.0)) for e in bwQueryNeg])+"\n")

dnasePosFile.close()
dnaseNegFile.close()
dnaseTempFile.close()



#################################################################################################
# Apply centipede to defined (MPBS) regions on a set of signals and prior information.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the motif.")
params.append("    2. mpbsFileName = MPBS file name separated by evidence.")
params.append("    3. dampFileName = CENTIPEDE parameter file.")
params.append("    4. priorFileName = File containing priors in tab separated format.")
params.append("    5. dnaseFileName = File containing dnase counts in tab separated format.")
params.append("    6. outputFileName = Location + name of output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputfileName> = Bed file containing MPBSs with centipede posterior.")
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
#dampFileName = sys.argv[3]
priorFileName = sys.argv[3]
dnaseFileName = sys.argv[4]
outputFileName = sys.argv[5]

# Parameters
tempFiles = [] # Temporary files list to erase at the end

# Reading mpbs list
mpbsFile = open(mpbsFileName,"r")
mpbsList = []
for line in mpbsFile:
    ll = line.strip().split("\t")
    if(ll[3].split(":")[0] == mpbsName): mpbsList.append(ll)
mpbsFile.close()

# Reading damp parameters
"""
dampFile = open(dampFileName,"r")
dampFile.readline() # header
dampLambda = 0.0; dampNegBin = 0.0
for line in dampFile:
    ll = line.strip().split("\t")
    if(ll[0] == mpbsName):
        dampLambda = ll[1]
        dampNegBin = ll[2]
        break
dampFile.close()
"""
dampLambda = "0.75"
dampNegBin = "0.0"

# Applying CENTIPEDE
centipedeFileName = outputFileName+".CENTIPEDE_TEMP.R"
centipedeOutputFileName = outputFileName+".CENTIPEDE_TEMP.Rout"
centipedePosteriorFileName = outputFileName+".CENTIPEDE_TEMP.txt"
logLikeFileName = outputFileName+".CENTIPEDE_LL.txt"
tempFiles.append(centipedeFileName)
tempFiles.append(centipedeOutputFileName)
tempFiles.append(centipedePosteriorFileName)
tempFiles.append(logLikeFileName)
centipedeFile = open(centipedeFileName,"w")
#centipedeFile.write("library(\"CENTIPEDE\", lib.loc=\"/hpcwork/izkf/app/CENTIPEDE/\")\n")
centipedeFile.write("library(CENTIPEDE)\n")
centipedeFile.write("priorSig <- as.matrix(read.table(\""+priorFileName+"\", sep=\"\\t\", header=FALSE))\n")
rListSignals = "Xlist = list("
rListSignals += "dnaseFPSig,"
centipedeFile.write("dnaseFPSig <- as.matrix(read.table(\""+dnaseFileName+"\", sep=\"\\t\", header=FALSE))\n")
"""
if(useTotCounts):
    rListSignals += "dnaseCOSig,"
    centipedeFile.write("dnaseCOSig <- as.matrix(read.table(\""+dnaseCountTempFileName+"\", sep=\"\\t\", header=FALSE))\n")
if(activeHistList != ["."]):
    rListSignals += "activSig,"
    centipedeFile.write("activSig <- as.matrix(read.table(\""+aHistTempFileName+"\", sep=\"\\t\", header=FALSE))\n")
if(represHistList != ["."]):
    rListSignals += "represSig,"
    centipedeFile.write("represSig <- as.matrix(read.table(\""+rHistTempFileName+"\", sep=\"\\t\", header=FALSE))\n")
"""
rListSignals = rListSignals[:-1]+")"
centipedeFile.write("centFit <- fitCentipede("+rListSignals+", Y=priorSig, DampLambda="+dampLambda+", DampNegBin="+dampNegBin+")\n")
centipedeFile.write("write(centFit$PostPr, file = \""+centipedePosteriorFileName+"\", ncolumns = 1, append = FALSE)\n")
centipedeFile.write("write(centFit$LogLikEnd, file = \""+logLikeFileName+"\", ncolumns = 1, append = FALSE)\n")
centipedeFile.close()
os.system("R CMD BATCH "+centipedeFileName+" "+centipedeOutputFileName)

# Reading CENTIPEDE results
posteriorList = []
centipedePosteriorFile = open(centipedePosteriorFileName,"r")
for line in centipedePosteriorFile: posteriorList.append(float(line.strip()))
centipedePosteriorFile.close()

# Writing CENTIPEDE results
outputFile = open(outputFileName,"w")
for i in range(0,len(mpbsList)):
    mpbsList[i][4] = str(posteriorList[i])
    outputFile.write("\t".join(mpbsList[i])+"\n")
outputFile.close()

# Termination
for e in tempFiles: os.system("rm "+e)



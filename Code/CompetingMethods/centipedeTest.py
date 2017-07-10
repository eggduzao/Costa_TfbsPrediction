#################################################################################################
# Test on centipede parameters
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the tested MPBS.")
params.append("    2. consFileName = Conservation file.")
params.append("    3. tssFileName = TSS file.")
params.append("    4. dnasePosFileName = DNase counts on positive strand file.")
params.append("    5. dnaseNegFileName = DNase counts on negative strand file.")
params.append("    6. mpbsFileName = MPBS file.")
params.append("    7. outputDataLoc = Output data files.")
params.append("    8. outputResLoc = Output result files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputResLoc>L_<lambda>_N_<negbin>.bed = Result mpbs with posteriors.")
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
consFileName = sys.argv[2]
tssFileName = sys.argv[3]
dnasePosFileName = sys.argv[4]
dnaseNegFileName = sys.argv[5]
mpbsFileName = sys.argv[6]
outputDataLoc = sys.argv[7]
outputResLoc = sys.argv[8]

# Damp lists
dnaseTotExt = 200
smallerTest = True
smallerTestMax = 1000
dampLambdaList = [0.0,0.25,0.5,0.75,1.0]
dampNegBinList = [0.0,0.25,0.5,0.75,1.0]

#############################################################################################
### MPBS
#############################################################################################

# Reading mpbs list
mpbsFile = open(mpbsFileName,"r")
mpbsList = []
if(smallerTest):
    posCount = 0
    negCount = 0
    for line in mpbsFile:
        ll = line.strip().split("\t")
        if(ll[3].split(":")[0] == mpbsName):
            if(ll[3].split(":")[1] == "Y"):
                if(posCount >= smallerTestMax): continue
                else: posCount += 1
            if(ll[3].split(":")[1] == "N"):
                if(negCount >= smallerTestMax): continue
                else: negCount += 1
            mpbsList.append(ll)
else:
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
outputPriorFileName = outputDataLoc+mpbsName+"_PRIOR.txt"
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
dnaseTempFileName = outputDataLoc+mpbsName+"_DNASE.txt"
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

    # Inverting minus strand sequences
    #if(strand == "-"):
    #    bwQueryPos = bwQueryPos[::-1]
    #    bwQueryNeg = bwQueryNeg[::-1]

    # Limiting counts to 30bp
    bwQueryPos = [min(e,30.0) for e in bwQueryPos]
    bwQueryNeg = [min(e,30.0) for e in bwQueryNeg]

    # Writing footprint signal
    dnaseTempFile.write("\t".join([str(e) for e in bwQueryPos])+"\t"+"\t".join([str(e) for e in bwQueryNeg])+"\n")

# Closing files
dnasePosFile.close()
dnaseNegFile.close()
dnaseTempFile.close()

#############################################################################################
### CENTIPEDE
#############################################################################################

# Opening centipede file
centFileName = outputResLoc+"cent.R"
centFile = open(centFileName,"w")

# Reading data
centFile.write("library(\"CENTIPEDE\", lib.loc=\"/hpcwork/izkf/app/CENTIPEDE/\")\n")
centFile.write("priorSig <- as.matrix(read.table(\""+outputPriorFileName+"\", sep=\"\\t\", header=FALSE))\n")
centFile.write("dnaseFPSig <- as.matrix(read.table(\""+dnaseTempFileName+"\", sep=\"\\t\", header=FALSE))\n")

# centFit function
for dampLambda in dampLambdaList:
    for dampNegBin in dampNegBinList:
        postFileName = outputResLoc+"L_"+str(dampLambda)+"_N_"+str(dampNegBin)+".post"
        centFile.write("centFit <- fitCentipede(Xlist = list(dnaseFPSig), Y=priorSig, DampLambda="+str(dampLambda)+", DampNegBin="+str(dampNegBin)+")\n")
        centFile.write("write(centFit$PostPr, file = \""+postFileName+"\", ncolumns = 1, append = FALSE)\n")

# Executing CENTIPEDE
centFile.close()
centFileOutName = outputResLoc+"cent.Rout"
os.system("R CMD BATCH "+centFileName+" "+centFileOutName)

# Creating bed with posterior
for dampLambda in dampLambdaList:
    for dampNegBin in dampNegBinList:

        # Output files
        postFileName = outputResLoc+"L_"+str(dampLambda)+"_N_"+str(dampNegBin)+".post"
        outFileName = outputResLoc+"L_"+str(dampLambda)+"_N_"+str(dampNegBin)+".bed"

        # Reading posterior
        posteriorList = []
        centipedePosteriorFile = open(postFileName,"r")
        for line in centipedePosteriorFile: posteriorList.append(float(line.strip()))
        centipedePosteriorFile.close()        

        # Writing results
        outputFile = open(outFileName,"w")
        for i in range(0,len(mpbsList)):
            mpbsList[i][4] = str(posteriorList[i])
            outputFile.write("\t".join(mpbsList[i])+"\n")
        outputFile.close()

        # Removing temporary posterior
        os.system("rm "+postFileName)

# Termination
os.system("rm "+centFileName)
os.system("rm "+centFileOutName)



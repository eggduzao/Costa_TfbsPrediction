#################################################################################################
# Applies Cuellar-Partida algorithm and finds motifs for a particular factor using epigenetic
# priors.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the MPBS.")
params.append("    2. fimoThresh = Threshold for FIMO.")
params.append("    3. coordFileName = Bed file with the regions to apply MM.")
params.append("    4. memeFileName = PWM file name in meme format.")
params.append("    5. priorFileNameList = List of prior files separated by chromosome.")
params.append("    6. genomeLocation = Location of the genome fasta files.")
params.append("    7. outputLocation = Location of the output file.")
params.append("    8. outputFolder = Temporary folder.")
params.append("###")
params.append("Output: ")
params.append("    1. <signalName>_<mpbsName>_cuellar.bed")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsName = sys.argv[1]
fimoThresh = sys.argv[2]
coordFileName = sys.argv[3]
memeFileName = sys.argv[4]
priorFileNameList = sys.argv[5].split(",")
genomeLocation = sys.argv[6]
if(genomeLocation[-1] != "/"): genomeLocation+="/"
outputLocation = sys.argv[7]
if(outputLocation[-1] != "/"): outputLocation+="/"
outputFolder = sys.argv[8]
if(outputFolder[-1] != "/"): outputFolder+="/"

# File names
signalName = "".join([e.split("/")[-1].split(".")[0] for e in priorFileNameList])

# Fetch motif length
pwmFile = open(memeFileName,"r")
for line in pwmFile:
    if("w=" in line):
        ll = line.strip().split(" ")
        motifLen = ll[5]
        break
pwmFile.close()

# Initialization
coordFile = open(coordFileName,"r")
outputFile = open(outputLocation+mpbsName+".bed","w")
priorFileList = [open(e,"r") for e in priorFileNameList]
bwList = [BigWigFile(e) for e in priorFileList]

# Iterating on coordFile
for line in coordFile:

    # Initializations
    ll = line.strip().split("\t")
    chrName = ll[0]; c1 = int(ll[1]); c2 = int(ll[2])

    # Creating psp file
    minValue = 999.9
    for i in range(0,len(bwList)):

        # Creating prior file
        sigVec = aux.correctBW(bwList[i].get(chrName,c1,c2),c1,c2)
        minV = min(sigVec)
        if(minV < minValue): minValue = minV
        priorFileName = outputFolder+chrName+"_"+str(i)+".prior"
        pspFileName = outputFolder+chrName+"_"+str(i)+".psp"
        priorFile = open(priorFileName,"w")
        priorFile.write("variableStep chrom="+chrName+" span=1 min_val="+str(minV)+"\n")
        for v in range(0,len(sigVec)):
            priorFile.write(str(c1+v)+" "+str(sigVec[v])+"\n")
        priorFile.close()

        # Creating psp file from prior
        os.system("to_psp -a "+str(c1)+" -b "+str(c2)+" -i "+priorFileName+" -o "+pspFileName+" -m "+motifLen+" &> /dev/null")
        os.system("sed -i '$a\\' "+pspFileName)
    pspFinalFileName =  outputFolder+chrName+".psp"
    os.system("cat "+outputFolder+chrName+"_*.psp > "+pspFinalFileName)

    # Creating prior dist
    distFileName1 = outputFolder+chrName+".dist"
    os.system("compute-prior-dist 100 "+pspFinalFileName+" > "+distFileName1)
    
    # Fetching sequence
    bedFileName = outputFolder+chrName+".bed"
    fastFileName = outputFolder+chrName+".fa"
    bedFile = open(bedFileName,"w")
    bedFile.write("\t".join([chrName,str(c1),str(c2)])+"\n")
    bedFile.close()
    os.system("fastaFromBed -fi "+genomeLocation+chrName+".fa -bed "+bedFileName+" -fo "+fastFileName)

    # Applying FIMO
    resFileName = outputFolder+chrName+".txt"
    os.system("fimo --verbosity 1 --thresh "+fimoThresh+" --psp "+pspFinalFileName+" --prior-dist "+distFileName1+" --text "+memeFileName+" "+fastFileName+" > "+resFileName+" 2> /dev/null")

    # Writing bed output
    fimoFile = open(resFileName,"r")
    fimoFile.readline()
    for line in fimoFile:
        ll = line.strip().split("\t")
        outputFile.write("\t".join([ll[1],ll[2],str(int(ll[3])+1),signalName+":"+mpbsName,ll[5],ll[4]])+"\n")
    fimoFile.close()

# Termination
outputFile.close()



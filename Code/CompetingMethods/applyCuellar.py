#################################################################################################
# Applies Cuellar-Partida algorithm and finds motifs for a particular factor using epigenetic
# priors.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the MPBS.")
params.append("    2. fimoThresh = Threshold for FIMO.")
params.append("    3. memeFileName = PWM file name in meme format.")
params.append("    4. priorFileNameList = List of prior files separated by chromosome.")
params.append("    5. genomeLocation = Location of the genome fasta files.")
params.append("    6. outputLocation = Location of the output file.")
params.append("    7. outputFolder = Temporary folder.")
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
memeFileName = sys.argv[3]
priorFileNameList = sys.argv[4].split(",")
genomeLocation = sys.argv[5]
if(genomeLocation[-1] != "/"): genomeLocation+="/"
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"
outputFolder = sys.argv[7]
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

# Fetch chrom sizes
chromSizesDict = dict()
chromSizesFile = open("/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered","r")
for line in chromSizesFile:
    ll = line.strip().split("\t")
    chromSizesDict[ll[0]] = int(ll[1])
chromSizesFile.close()

# Iterating on each chromosome
chrList = constants.getChromList(x=True, y=False)
outputFile = open(outputLocation+mpbsName+".bed","w")
for chrName in chrList:

    # Initializations
    priorFileList = [open(e,"r") for e in priorFileNameList]
    bwList = [BigWigFile(e) for e in priorFileList]

    # Fetching first and last positions
    p1 = -1; p2 = 9999999999; ext = 10000; chromSize = chromSizesDict[chrName]
    for bw in bwList:

        # First position
        firstQ = aux.correctBW(bw.get(chrName,0,ext),0,ext)
        currP1 = 0
        for e in firstQ:
            currP1+=1
            if(e > 0.0): break
        if(currP1 > p1): p1 = currP1
        
        # Last position
        lastQ = aux.correctBW(bw.get(chrName,chromSize-ext,chromSize+ext),chromSize-ext,chromSize+ext)
        currP2 = chromSize-ext
        for e in lastQ:
            if(e == 0.0): break
            currP2+=1
        if(currP2 < p2): p2 = currP2

    # Iterating on genomic positions
    window = 1000000
    for k in range(p1,p2,window):

        # Initializations
        toDelete = []
        c1 = k; c2 = min(k+window,p2)

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
            toDelete.append(priorFileName); toDelete.append(pspFileName)
        pspFinalFileName =  outputFolder+chrName+".psp"
        os.system("cat "+outputFolder+chrName+"_*.psp > "+pspFinalFileName)
        toDelete.append(pspFinalFileName)

        # Creating prior dist
        #distFileName1 = outputFolder+chrName+".dist1"
        distFileName1 = outputFolder+chrName+".dist"
        os.system("compute-prior-dist 100 "+pspFinalFileName+" > "+distFileName1)
        toDelete.append(distFileName1)

        # Fixing minimum value of prior dist
        #distFileName2 = outputFolder+chrName+".dist"
        #os.system("sed 1d "+distFileName1+" > "+distFileName2)
        #os.system("sed -i -e '1i"+str(minValue)+"\\' "+distFileName2)
        #toDelete.append(distFileName2)
    
        # Fetching sequence
        bedFileName = outputFolder+chrName+".bed"
        fastFileName = outputFolder+chrName+".fa"
        bedFile = open(bedFileName,"w")
        bedFile.write("\t".join([chrName,str(c1),str(c2)])+"\n")
        bedFile.close()
        os.system("fastaFromBed -fi "+genomeLocation+chrName+".fa -bed "+bedFileName+" -fo "+fastFileName)
        toDelete.append(bedFileName); toDelete.append(fastFileName)

        # Applying FIMO
        resFileName = outputFolder+chrName+".txt"
        os.system("fimo --verbosity 1 --thresh "+fimoThresh+" --psp "+pspFinalFileName+" --prior-dist "+distFileName1+" --text "+memeFileName+" "+fastFileName+" > "+resFileName+" 2> /dev/null")
        toDelete.append(resFileName)

        # Writing bed output
        fimoFile = open(resFileName,"r")
        fimoFile.readline()
        for line in fimoFile:
            ll = line.strip().split("\t")
            outputFile.write("\t".join([ll[1],ll[2],str(int(ll[3])+1),signalName+":"+mpbsName,ll[5],ll[4]])+"\n")
        fimoFile.close()

        # Removing files
        #for e in toDelete: os.system("rm "+e)

# Termination
outputFile.close()
#os.system("rm -rf "+outputFolder)



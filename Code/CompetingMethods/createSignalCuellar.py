#################################################################################################
# Script to create the signal used by Cuellar method.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. minWindow = Length of the wig step (whole).")
params.append("    2. maxWindow = Length of the window to count the reads (half of extension).")
params.append("    3. quartWindow = Length of the window whose reads worth 0.25 (half of ext.).")
params.append("    4. chromSizesFileName = Chrom sizes file name.")
params.append("    5. bamFileName = Bam file to create the signal.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <bamFileName>.bw = Resulting signal.")
params.append("###")
#################################################################################################

# Import
import os
import sys
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
minWindow = int(sys.argv[1])
maxWindow = int(sys.argv[2])
quartWindow = int(sys.argv[3])
chromSizesFileName = sys.argv[4]
bamFileName = sys.argv[5]
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Output name
outputName = bamFileName.split("/")[-1].split(".")[0]

# Fetching chrom sizes
chromSizesFile = open(chromSizesFileName,"r")
chromSizesDict = dict()
for line in chromSizesFile:
    ll = line.strip().split("\t")
    chromSizesDict[ll[0]] = int(ll[1])
chromSizesFile.close()

# Creating bed files
genomeFileName = outputLocation+outputName+"_genome.bed"
genomeFile = open(genomeFileName,"w")
if(quartWindow > 0):
    quartFileName = outputLocation+outputName+"_quart.bed"
    quartFile = open(quartFileName,"w")
chrList = constants.getChromList(x=True, y=False)
totHalfExt = maxWindow + quartWindow
for chrName in chrList:
    for i in xrange(totHalfExt,chromSizesDict[chrName]-minWindow-maxWindow-quartWindow,minWindow):
        g1 = i-maxWindow; g2 = i+minWindow+maxWindow
        genomeFile.write("\t".join([chrName,str(g1),str(g2)])+"\n")
        if(quartWindow > 0):
            q1_1 = i-maxWindow-quartWindow; q1_2 = i-maxWindow
            q2_1 = i+minWindow+maxWindow; q2_2 = i+minWindow+maxWindow+quartWindow
            quartFile.write("\t".join([chrName,str(q1_1),str(q1_2)])+"\n"+"\t".join([chrName,str(q2_1),str(q2_2)])+"\n")
genomeFile.close()
if(quartWindow > 0): quartFile.close()

# Calculating coverage
genomeResFileNameTemp = outputLocation+outputName+"_genomerestemp.bed"
genomeResFileName = outputLocation+outputName+"_genomeres.bed"
os.system("coverageBed -counts -abam "+bamFileName+" -b "+genomeFileName+" > "+genomeResFileNameTemp)
os.system("sort -k1,1 -k2,2n "+genomeResFileNameTemp+" > "+genomeResFileName)
os.system("rm "+genomeResFileNameTemp)
if(quartWindow > 0): 
    quartResFileNameTemp = outputLocation+outputName+"_quartrestemp.bed"
    quartResFileName = outputLocation+outputName+"_quartres.bed"
    os.system("coverageBed -counts -abam "+bamFileName+" -b "+quartFileName+" > "+quartResFileNameTemp)
    os.system("sort -k1,1 -k2,2n "+quartResFileNameTemp+" > "+quartResFileName)
    os.system("rm "+quartResFileNameTemp)

# Converting bedtools results to wig
genomeResFile = open(genomeResFileName,"r")
if(quartWindow > 0): quartResFile = open(quartResFileName,"r")
outputFileName = outputLocation+outputName+".wig"
outputFile = open(outputFileName,"w")
lineG = genomeResFile.readline()
currChr = ""
while lineG:
    llG = lineG.strip().split("\t")
    if(quartWindow > 0):
        lineQ1 = quartResFile.readline(); llQ1 = lineQ1.strip().split("\t")
        lineQ2 = quartResFile.readline(); llQ2 = lineQ2.strip().split("\t")
    if(llG[0] != currChr):
        currChr = llG[0]
        outputFile.write("fixedStep chrom="+currChr+" start="+str(int(llG[1])+maxWindow+1)+" step="+str(minWindow)+" span="+str(minWindow)+"\n")
    value = float(llG[3])
    if(quartWindow > 0):
        value += ((float(llQ1[3])+float(llQ2[3]))*0.25)
    outputFile.write(str(value)+"\n")
    lineG = genomeResFile.readline()
genomeResFile.close()
if(quartWindow > 0): quartResFile.close()
outputFile.close()

# Convert wig to bigwig
os.system("wigToBigWig "+outputFileName+" "+chromSizesFileName+" "+outputFileName[:-3]+"bw")
os.system("rm "+outputFileName)

# Termination
os.system("rm "+genomeFileName+" "+genomeResFileName)
if(quartWindow > 0): os.system("rm "+quartFileName+" "+quartResFileName)



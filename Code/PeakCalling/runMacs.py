#################################################################################################
# Runs MACS 1.4 program keeping the necessary files (counts) and discarding log files.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. shiftSize = The number of bp to extend the tags.")
params.append("                   Use 1 for DNase-seq and 100 for ChIP-seq.")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("    3. [folderList] = List of folders containing .bed files (one for each chrom.")
params.append("       i.e. output of sepRepByChr.py). Each folder represents one cell line.")
params.append("###")
params.append("Output: ")
params.append("    1. [<chrName>_counts.wig] = List of output WIG files (per chromossome).")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
shiftSize = int(sys.argv[1])
outputLocation = sys.argv[2]
if(outputLocation[-1] != "/"): outputLocation+="/"
folderList = sys.argv[3:]
for i in range(0,len(folderList)):
    if(folderList[i][-1] != "/"): folderList[i]+="/"

# Merging all input files from all input folders
mergeCmd = "bedMerge "+outputLocation+"merged.bed"
for e in folderList: mergeCmd += " "+e+"*"
os.system(mergeCmd)

# Transforming bed to bam
os.system("bedToBam -i "+outputLocation+"merged.bed -g "+constants.getChromSizesLocation()+" > "+outputLocation+"merged.bam" )

# Running MACS
macsCmd = "macs14"
macsCmd += " -t "+outputLocation+"merged.bam"
macsCmd += " -f BAM"
macsCmd += " -g hs"
macsCmd += " -n TEMPTODELETE"
macsCmd += " -w"
macsCmd += " --space=1"
macsCmd += " --verbose=0"
macsCmd += " --shiftsize="+str(shiftSize)
os.system(macsCmd)

# Deleting possibly old files
for fileName in os.listdir(outputLocation):
    if("_counts" in fileName): os.system("rm "+outputLocation+fileName)

# Changing variableStep entries to fixedStep
wigLocation = outputLocation+"TEMPTODELETE_MACS_wiggle/treat/"
os.system("gunzip "+wigLocation+"*.gz")
for chrName in constants.getChromList():
    fileList = glob.glob(wigLocation+"*"+chrName+".wig")
    if(len(fileList) == 1):
        inputFile = open(fileList[0],"r")
        outputFileName = outputLocation+chrName+"_counts.wig"
        outputFile = open(outputFileName,"w")
        inputFile.readline(); line = inputFile.readline()
        if(line and line[0] == "v"): line = inputFile.readline()
        if(line): outputFile.write("fixedStep chrom="+chrName+" start="+line.strip().split("\t")[0]+" step=1\n")
        while line:
            outputFile.write(line.strip().split("\t")[1]+"\n")
            line = inputFile.readline()
        inputFile.close()
        outputFile.close()
        aux.wigToBigWig(outputFileName,outputFileName[:-3]+"bw",removeWig=False)

# Termination - Removing unecessary files
os.system("rm -rf "+outputLocation+"TEMPTODELETE* "+outputLocation+"merged.bed "+outputLocation+"merged.bam")



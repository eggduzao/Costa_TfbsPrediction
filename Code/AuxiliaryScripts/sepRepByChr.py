#################################################################################################
# Separates a set of .bed files representing replicates of a genome-wide study into a set of bed 
# files (one per chromosome) which contain all replicates merged.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. separator = The separator in the bed file. Can be \"tab\" or \"spc\".")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("    3. [inputList] = List of .bed files (one for each replicate).")
params.append("###")
params.append("Output: ")
params.append("    1. [<chrName>_raw.bed] = List of files per chrom. with replicates merged.")
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
separator = sys.argv[1]
if(separator == "tab"): separator = "\t"
else: separator = " "
outputLocation = sys.argv[2]
if(outputLocation[-1] != "/"): outputLocation+="/"
inputList = sys.argv[3:]

# File handling - creating all chromossomes files to append data later
chrNames = constants.getChromList()
written = [False] * len(chrNames)
for e in chrNames:
    initialFile = open(outputLocation+e+"_raw.bed","w")
    initialFile.close()

# Iterate on input file list
outputFile = None
for inputFileName in inputList:

    # Open input file and get first line
    inputFile = open(inputFileName,"r")
    firstLine = inputFile.readline()
    currChr = firstLine.strip().split(separator)[0]

    # Create output file on current chromossome
    if(currChr in chrNames):
        outputFile = open(outputLocation+currChr+"_raw.bed","a")
        outputFile.write(firstLine)
        written[chrNames.index(currChr)] = True

    # Iterate on input file
    for line in inputFile:
        if(len(line) < 2): continue
        lineList = line.strip().split(separator)
        if(lineList[0] == currChr): 
            outputFile.write(line)
        elif(lineList[0] in chrNames):
            outputFile.close()
            currChr = lineList[0]
            written[chrNames.index(currChr)] = True
            outputFile = open(outputLocation+currChr+"_raw.bed","a")
            outputFile.write(line)

    inputFile.close()
    if(outputFile): outputFile.close()
# End for inputFileName in inputList

# Termination - deleting chromossome files that have not been written
for i in range(0,len(written)):
    if(not written[i]):
        os.remove(outputLocation+chrNames[i]+"_raw.bed")



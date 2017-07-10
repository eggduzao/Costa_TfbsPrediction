#################################################################################################
# Separates a set of .bed files with mixed chromosomes into a set of bed files per chromosome.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. separator = The separator in the bed file. Can be \"tab\" or \"spc\".")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("    3. [inputList] = List of .bed files (one for each replicate).")
params.append("###")
params.append("Output: ")
params.append("    1. [<chrName>.bed] = List of files per chrom. with replicates merged.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import operator
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

# Initializing coordDict
chromSet = constants.getChromList(y=False)
coordDict = dict([(e,[]) for e in chromSet])
for inputFile in inputList:
    tempDict = aux.createBedDictFromSingleFile(inputFile, separator=separator)
    for e in chromSet:
        if(e in tempDict.keys()): 
            for v in tempDict[e]: coordDict[e].append(v)

# Sorting coordDict
coordDict = gsort.sortBedDictionaries([coordDict])[0]

# Verifying input extension
ext = inputList[0].split(".")[-1]

# Iterating on chromSet
for chrom in chromSet:

    # Opening output file
    outputFile = open(outputLocation+chrom+"."+ext,"w")

    # Iterating on dictList
    for vec in coordDict[chrom]: outputFile.write(chrom+" "+" ".join([str(e) for e in vec])+"\n")

    # Closing output file
    outputFile.close()

# End for chrom in chromSet

# Removing empty files
for chrom in chromSet:
    inFileName = outputLocation+chrom+"."+ext
    inFile = open(inFileName,"r")
    line = inFile.readline()
    inFile.close()
    if(not line): os.system("rm "+inFileName)



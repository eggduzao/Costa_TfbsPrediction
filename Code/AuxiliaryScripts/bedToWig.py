#################################################################################################
# Converts a bed file to a wig file.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. field = Field with the desired value to be converted to a wig signal.")
params.append("    2. separator = Separator character on bed file (tab or spc).")
params.append("    3. inputBedName = The location + name of the input bed file.")
params.append("    4. outputWigName = The location + name of the output wig file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputWigName>.wig = Resulting wig file.")
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

# Input
field = int(sys.argv[1])
separator = sys.argv[2]
if(separator == "tab"): separator = "\t"
elif(separator == "spc"): separator = " "
inputBedName = sys.argv[3]
outputWigName = sys.argv[4]

# Reading and sorting input
bedDict = aux.createBedDictFromSingleFile(inputBedName, separator=separator)
bedDict = gsort.sortBedDictionaries([bedDict], field=0)[0]

# Writing wig
outputFile = open(outputWigName,"w")
chrList = constants.getChromList(reference=[bedDict])
for chrName in chrList:
    currPos = -1
    for e in bedDict[chrName]:
        if(currPos != e[0]):
            currPos = e[1]
            outputFile.write("fixedStep chrom="+chrName+" start="+str(e[0]+1)+" step=1\n")
            for k in range(e[0],e[1]): outputFile.write(str(e[field-1])+"\n")
        else:
            currPos = e[1]
            for k in range(e[0],e[1]): outputFile.write(str(e[field-1])+"\n")
outputFile.close()



#################################################################################################
# Sums wig files.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. outputFileName = Location and name of the output file.")
params.append("    2. [wigList] = List of wig files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName>.wig = Summed file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import operator
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
outputFileName = sys.argv[1]
wigList = sys.argv[2:]

# File Handling
outputFile = open(outputFileName,"w")

# Verifying input type
inType = wigList[0].split(".")[-1]

# Populating vector with firstFile
if(inType == "bw"): 
    aux.bigWigToWig(wigList[0],wigList[0][:-2]+".wig")
    wigFile = open(wigList[0][:-2]+".wig","r")
else:
    wigFile = open(wigList[0],"r")
dataVec = [0.0] * (int(wigFile.readline().split(" ")[2][6:])-1)
for line in wigFile:
    if(line[0] == "f"): continue
    dataVec.append(float(line.strip()))
wigFile.close()
if(inType == "bw"): os.remove(wigList[0][:-2]+".wig")

# Iterating on wigList
for wigFileName in wigList[1:]:

    # Summing to vector
    if(inType == "bw"): 
        aux.bigWigToWig(wigFileName,wigFileName[:-2]+".wig")
        wigFile = open(wigFileName[:-2]+".wig","r")
    else:
        wigFile = open(wigFileName,"r")
    counter = int(wigFile.readline().split(" ")[2][6:])-1
    for line in wigFile:
        if(line[0] == "f"): continue
        if(counter < len(dataVec)): dataVec[counter] += float(line.strip())
        else: dataVec.append(float(line.strip()))
        counter += 1
    wigFile.close()
    if(inType == "bw"): os.remove(wigFileName[:-2]+".wig")

# End for wigFileName in wigList[1:]

# Verifying first non-0 entry
for firstIndex in range(0,len(dataVec)):
    if(dataVec[firstIndex] > 0): break

# Writing data
outputFile.write("fixedStep chrom=chr6 start="+str(firstIndex+1)+" step=1\n")
for v in dataVec[firstIndex:]: outputFile.write(str(v)+"\n")
outputFile.close()

# Converting wig to bigWig
aux.wigToBigWig(outputFileName,outputFileName[:-3]+"bw",removeWig=False)



#################################################################################################
# Converts a homer file to a jaspar file.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. inputFileName = File containing all the homer motifs to be converted.")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputLocation><factorName>.pwm = Resulting factors.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
inputFileName = sys.argv[1]
outputLocation = sys.argv[2]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Converting motifs
inputFile = open(inputFileName,"r")
flagFirst = True; nuc = ["A","C","G","T"]
motDic = dict([(e,[]) for e in nuc])
for line in inputFile:
    ll = line.strip().split("\t")
    if(line[0] == ">"):
        if(flagFirst):
            outputFile = open(outputLocation+"_".join(ll[1].split("/"))+".pwm","w")
        else:
            for k in nuc: outputFile.write(" ".join(motDic[k])+"\n")
            outputFile.close()
            motDic = dict([(e,[]) for e in nuc])
    else:
        for i in range(0,len(nuc)): motDic[nuc[i]].append(ll[i])
for k in nuc: outputFile.write(" ".join(motDic[k])+"\n")
outputFile.close()

# Termination
inputFile.close()



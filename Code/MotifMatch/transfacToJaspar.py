#################################################################################################
# Converts a PWM file in transfac format into jaspar format.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. inputFileName = Location of the transfac formated file.")
params.append("    2. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>.pfm = The jaspar formated file.")
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

# File handling
inputFile = open(inputFileName,"r")
outputFile = open(outputLocation+inputFileName.split("/")[-1].split(".")[0]+".pfm","w")

# Converting transfac into jaspar
values = [[],[],[],[]]
flagStart = False
for line in inputFile:
    if(not flagStart):
        if(line[0] == "P"):
            flagStart = True
            continue
        elif(line[0] == "0"):
            flagStart = True
    if(flagStart):
        if(line[0] == "X"): break
        if("\t" in line): ll = filter(None,line.strip().split("\t"))
        else: ll = filter(None,line.strip().split(" "))
        for i in range(0,len(values)): values[i].append(ll[i+1])
# End for line in fileTrans

# Writing into file
for e in values: outputFile.write(" ".join(e)+"\n")

# Termination
inputFile.close()
outputFile.close()



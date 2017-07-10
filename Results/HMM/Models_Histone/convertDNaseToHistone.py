# This script convert readable annotations based on DNase+Histone to
# readable annotations based on histones only. This is performed by
# considering every DNase level state + FP = histone FP. The new
# HMM has only 5 states.

# Import
import os
import sys
import glob

# Input
inputFileList = glob.glob("./readable/H3K9ac_proximal.txt")

# Iterating on each input list
for inputFileName in inputFileList:

    # Copying file
    os.system("cat "+inputFileName+" > "+inputFileName+"TEMP")
 
    # Writing new file
    inFile = open(inputFileName+"TEMP","r")
    outFile = open(inputFileName,"w")
    outFile.write("\n# Coordinates are 1-relative (Genome browser and wig files)\n0 = BACK\n1 = UPH\n2 = TOPH\n3 = DOWNH\n4 = FP\n\n")
    for e in range(0,11): inFile.readline()
    for line in inFile:
        if(len(line) < 2):
            outFile.write("\n")
            continue
        ll = line.strip().split(" ")
        for i in range(1,5):
            newLL = ""
            for j in range(0,len(ll[i])):
                if(ll[i][j] in ["5","6","7"]): newLL+="4"
                else: newLL+=ll[i][j]
            ll[i] = newLL
        outFile.write(" ".join(ll)+"\n")
    
    # Termination
    inFile.close()
    outFile.close()
    os.system("rm "+inputFileName+"TEMP")



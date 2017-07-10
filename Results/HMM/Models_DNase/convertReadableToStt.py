
# Import
import os
import sys

# Iterating on cell types
cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562"]
for cell in cellList:

    inFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase.txt"
    outFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase.stt"
    inFile = open(inFileName,"r")
    outFile = open(outFileName,"w")

    pos1 = -1
    isFirst = True
    pos2 = -1
    seq = ""

    for line in inFile:
        if(len(line) < 2 or "#" in line or "=" in line): continue
        lineList = line.strip().split(" ")
        if(pos1 == -1): pos1 = int(lineList[0]) - 1
        pos2 = int(lineList[-1])
        lineList = lineList[1:len(lineList)-1]
        seq += "".join(lineList)
        
    outFile.write("chr1 2114"+str(pos1)+" 2114"+str(pos2)+" "+seq)

    inFile.close()
    outFile.close()



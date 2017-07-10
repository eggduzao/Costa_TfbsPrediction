
# Import
import os
import sys

# Parameters for new model
stateNameDict = dict([(0,"BACK"), (1,"UPD"), (2,"TOPD"), (3,"DOWND"), (4,"FP")])
colorDict = dict([(0,"50,50,50"), (1,"10,80,0"), (2,"20,40,150"), (3,"150,20,40"), (4,"198,150,0")])

# Iterating on cell types
cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562"]
for cell in cellList:

    inFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase.stt"
    outFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase.bed"
    inFile = open(inFileName,"r")
    outFile = open(outFileName,"w")
 
    for line in inFile:
        ll = line.strip().split(" ")
        stateList = [int(e) for e in ll[3]]
        currStart = int(ll[1])
        currPos = currStart + 1
        currV = stateList[0]
        for v in stateList[1:]:
            if(v != currV):
                outFile.write(ll[0]+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")
                currStart = currPos
                currV = v
            currPos += 1
        outFile.write(ll[0]+" "+str(currStart)+" "+str(currPos)+" "+stateNameDict[currV]+" 1000 . "+str(currStart)+" "+str(currPos)+" "+colorDict[currV]+"\n")

    inFile.close()
    outFile.close()



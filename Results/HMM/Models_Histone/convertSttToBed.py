
# Import
import os
import sys

# Input
outputLocation = sys.argv[1]
inputList = sys.argv[2:]

# Parameters for new model
stateNameDict = dict([(0,"BACK"), (1,"UPH"), (2,"TOPH"), (3,"DOWNH"), (4,"FP")])
colorDict = dict([(0,"50,50,50"), (1,"110,250,110"), (2,"90,180,240"), (3,"255,80,90"), (4,"198,150,0")])

# Converting
for fileName in inputList:

    inFile = open(fileName,"r")
    outFile = open(outputLocation+fileName.split("/")[-1].split(".")[0]+".bed","w")
 
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



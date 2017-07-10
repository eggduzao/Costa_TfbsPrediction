
# Script to convert a readable file into an STT file.
# Pass the file as argument.

import sys

outputLocation = sys.argv[1]
inList = sys.argv[2:]

for inFileName in inList:

    inFile = open(inFileName,"r")
    outFile = open(outputLocation+inFileName.split("/")[-1].split(".")[0]+".stt","w")

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
        
    outFile.write("chr1 "+str(pos1)+" "+str(pos2)+" "+seq)

    inFile.close()
    outFile.close()



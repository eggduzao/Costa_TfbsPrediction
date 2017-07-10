
import os
import sys

inList = [ "/home/egg/Desktop/ZBT7B_M00405.pfm", "/home/egg/Desktop/MA0138.2.REST.pfm", "/home/egg/Desktop/MA0527.1.ZBTB33.pfm" ]

for inFileName in inList:
  outFileName = inFileName[:-3]+"pwm"
  inName = inFileName.split("/")[-1].split(".")[0]
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")
  inTable = []
  for line in inFile:
    inTable.append(line.strip().split(" "))
  outFile.write("\t".join(["DE",inName,inName])+"\n")
  counter = 0
  for j in range(0,len(inTable[0])):
    outFile.write("\t".join([str(counter),inTable[0][j],inTable[1][j],inTable[2][j],inTable[3][j],"X"])+"\n")
    counter += 1
  outFile.write("XX")
  outFile.close()
  


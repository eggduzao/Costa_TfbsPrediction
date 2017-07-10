
# Import
import os
import sys

# Paramters
cellList = ["H1hesc","HeLaS3","HepG2","K562"]
inLoc="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/"
inFileList = ["DNase","H3K4me1","H3K4me3","Histones","DNaseHistones"]
genomeSize = 3300000000.0
outFile = open("result.txt","w")

# Cell line loop
for cell in cellList:

    # File Loop
    for inFileName in inFileList:

        inFile = open(inLoc+cell+"_Predictions/"+inFileName+".bed")
        summ = 0.0
        for line in inFile:
            ll = line.strip().split("\t")
            summ += (float(ll[2])-float(ll[1]))
        outFile.write("\t".join([cell,inFileName,str(int(summ)),str(round((summ/genomeSize)*100.0,2))])+"\n")
        inFile.close()
        
outFile.close()



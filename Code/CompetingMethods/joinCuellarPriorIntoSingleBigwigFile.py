# This script joins all the files inside the folders in folderList in a bigwig file

# Import
import os
import sys
import glob

# Input
inFolder = sys.argv[1]
chromSizes = sys.argv[2]

# Initializations
signalName = inFolder.split("/")[-2]
outLocation = "/".join(inFolder.split("/")[0:-2])+"/"
outFileName = outLocation+signalName+"_prior.wig"
outFile = open(outFileName,"w")

# Iterating on chromosomes
for inFileName in glob.glob(inFolder+"*.prior"):

    inFile = open(inFileName,"r")
    header = inFile.readline()
    outFile.write(" ".join(header.strip().split(" ")[:3])+"\n")
    for line in inFile: outFile.write(line)
    inFile.close()

# Converting to bigwig
outFile.close()
os.system("wigToBigWig "+outFileName+" "+chromSizes+" "+outFileName[:-3]+"bw")
os.system("rm "+outFileName)



###################################################################################################
# This script should be performed on all Boyle results.
###################################################################################################

# Import
import os
import sys 
import glob

# Input
inputFileName = sys.argv[1]

# Parameters
extList = [5]

###################################################################################################
# Initializations
###################################################################################################

# Initializations
inputLocation = "/".join(inputFileName.split("/")[:-1])+"/"
inputName = inputFileName.split("/")[-1].split(".")[0]
fpExtFileName = inputLocation+"Extension/"+inputName

###################################################################################################
# 1. Create extended footprints without merging
###################################################################################################

for ext in extList:

    # With merging
    outputFileName = fpExtFileName+"_"+str(ext)+".bed"
    os.system("extendAndMerge "+str(ext)+" "+str(ext)+" "+inputFileName+" "+outputFileName)

    """ Without merging
    inputFile = open(inputFileName,"r")
    outputFile = open(fpExtFileName+"_"+str(ext)+".bed","w")
    for line in inputFile:
        ll = line.strip().split("\t")
        p1 = str(max(int(ll[1])-ext,0)); p2 = str(int(ll[2])+ext)
        outputFile.write("\t".join([ll[0],p1,p2]+ll[3:])+"\n")
    inputFile.close()
    outputFile.close()
    """


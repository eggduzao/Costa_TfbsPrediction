###################################################################################################
# This script should be performed on all HMM results after they are ready.
###################################################################################################

# Import
import os
import sys 
import glob

# Input
inputFileName = sys.argv[1]
runNumber = sys.argv[2]

# First run with all HMM results inside HMM folders.
if(runNumber == "1"):

    # Parameters
    cellName = inputFileName.split("/")[-3]
    chromSizesFile = "/work/eg474423/eg474423_RegulatoryAnalysisTools/exp/script/chrom.sizes"
    extList = [5]

    ###################################################################################################
    # Initializations
    ###################################################################################################

    # Initializations
    outputLocation = "/".join(inputFileName.split("/")[:-2])+"/"
    outputLocationExt = outputLocation+"Extension/"
    fileName = inputFileName.split("/")[-1].split(".")[0]

    # Temporary and output file names
    fpBigBedFileName = inputFileName[:-3]+"bb"
    fpSpaceFileName = outputLocation+fileName+"_SPACE.bed"
    fpTabFileName = outputLocation+fileName+"_TAB.bed"
    fpTabSortFileName = outputLocation+fileName+".bed"
    fpExtFileName = outputLocationExt+fileName
    toBeRemoved = [fpSpaceFileName, fpTabFileName]

    ###################################################################################################
    # 1. Convert all bed results to bigbed
    ###################################################################################################

    #os.system("bedToBigBed "+inputFileName+" "+chromSizesFile+" "+fpBigBedFileName+" -verbose=0")

    ###################################################################################################
    # 2. Extract footprints
    ###################################################################################################

    os.system("cat "+inputFileName+" | grep FP > "+fpSpaceFileName) # Extract FP coordinates
    os.system("tr ' ' \\\\t < "+fpSpaceFileName+" > "+fpTabFileName) # Change space by tabs

    ###################################################################################################
    # 3. Sort FP file
    ###################################################################################################

    os.system("sort -k1,1 -k2,2n "+fpTabFileName+" > "+fpTabSortFileName) # Sorting file for intersectBed

    ###################################################################################################
    # 4. Create extended footprints without merging
    ###################################################################################################

    for ext in extList:
        inputFile = open(fpTabSortFileName,"r")
        outputFile = open(fpExtFileName+"_"+str(ext)+".bed","w")
        for line in inputFile:
            ll = line.strip().split("\t")
            p1 = str(max(int(ll[1])-ext,0)); p2 = str(int(ll[2])+ext)
            outputFile.write("\t".join([ll[0],p1,p2]+ll[3:])+"\n")
        inputFile.close()
        outputFile.close()

    ###################################################################################################
    # Termination
    ###################################################################################################

    for e in toBeRemoved: os.system("rm "+e)

# Second run with merged HMM results - The intention is only to create the extended footprints of these results
if(runNumber == "2"):

    # Parameters
    extList = [5]

    ###################################################################################################
    # Initializations
    ###################################################################################################

    inputLocation = "/".join(inputFileName.split("/")[:-1])+"/"
    inputName = inputFileName.split("/")[-1].split(".")[0]
    outputFileName = inputLocation+"Extension/"+inputName

    ###################################################################################################
    # 1. Create extended footprints without merging
    ###################################################################################################

    for ext in extList:
        inputFile = open(inputFileName,"r")
        outputFile = open(outputFileName+"_"+str(ext)+".bed","w")
        for line in inputFile:
            ll = line.strip().split("\t")
            p1 = str(max(int(ll[1])-ext,0)); p2 = str(int(ll[2])+ext)
            outputFile.write("\t".join([ll[0],p1,p2]+ll[3:])+"\n")
        inputFile.close()
        outputFile.close()



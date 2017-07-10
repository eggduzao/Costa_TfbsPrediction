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

#intOpt = "" # Original option in the paper
intOpt = " -u -f 1" # This option makes that only unique and complete overlaps are taken

# First run with all HMM results inside HMM folders.
if(runNumber == "1"):

    # Parameters
    cellName = inputFileName.split("/")[-3]
    labName = inputFileName.split("/")[-2].split("_")[2]
    wholeFolderName = inputFileName.split("/")[-2]
    if("m3134" in cellName): chromSizesFile = "/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes"
    else: chromSizesFile = "/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered.increased.10e6.ForBw"
    #peakToCutFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"+cellName+"/DNaseHistones.bed"
    if(labName == "DU"): peakToCutFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_DU/"+cellName+"/DNasePeaks.bed"
    elif(labName == "UW"): peakToCutFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNaseUW/"+cellName+"/DNasePeaks.bed"
    #if(labName == "DU"): peakToCutFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cellName+"/D13_DU.bed"
    #elif(labName == "UW"): peakToCutFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cellName+"/D13_UW.bed"
    #extList = [5,10]
    extList = []

    ###################################################################################################
    # Initializations
    ###################################################################################################

    # Initializations
    outputLocation = "/".join(inputFileName.split("/")[:-2])+"/"
    outputLocationExt = outputLocation+"Extension/"
    fileName = wholeFolderName+"_"+inputFileName.split("/")[-1].split(".")[0].split("_")[1]

    # Temporary and output file names
    fpBigBedFileName = inputFileName[:-3]+"bb"
    fpSpaceFileName = outputLocation+fileName+"_SPACE.bed"
    fpTabFileName = outputLocation+fileName+"_TAB.bed"
    fpTabSortFileName = outputLocation+fileName+"_TABSORT.bed"
    fpCutFileName = outputLocation+fileName+".bed"
    fpExtFileName = outputLocationExt+fileName
    toBeRemoved = [fpSpaceFileName, fpTabFileName, fpTabSortFileName]

    print fpCutFileName

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
    # 3. Cut HMM predictions by enriched regions (peaks)
    ###################################################################################################

    os.system("sort -k1,1 -k2,2n "+fpTabFileName+" > "+fpTabSortFileName) # Sorting file for intersectBed
    os.system("intersectBed -wa -a "+fpTabSortFileName+" -b "+peakToCutFileName+intOpt+" > "+fpCutFileName) # Cutting HMM by peak file

    ###################################################################################################
    # 4. Create extended footprints without merging
    ###################################################################################################

    for ext in extList:
        inputFile = open(fpCutFileName,"r")
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



###################################################################################################
# This script should be performed on all Cuellar results after they are ready.
###################################################################################################

# Import
import os
import sys 
import glob

###################################################################################################
# Input
###################################################################################################

# Input
inputLocation = sys.argv[1]

# Parameters
groupList = ["DNase,H3K4me1,H3K4me3,H3K9ac","DNase,H3K4me1,H3K4me3","DNase,H3K4me3,H3K9ac","DNase"] # It has to be in order of more signals to less signals
outputAlias = ["D139","D13","D39","D"] # Same size as groupList

###################################################################################################
# Format in order to remove _prior string and add cuellar type
###################################################################################################

# Fetching location and names
cuellarType = inputLocation.split("/")[-2]
outputLocation = "/".join(inputLocation.split("/")[:-2])+"/"

# Fetching files
fileNameList = glob.glob(inputLocation+"*.bed")

# Grouping files by signal type
groupDict = dict([(e,[]) for e in groupList])
for fileName in fileNameList:
    for k in groupList:
        allSignalList = k.split(",")
        flagBelongs = True
        for e in allSignalList:
            if(e not in fileName):
                flagBelongs = False
                break
        if(flagBelongs):
            groupDict[k].append(fileName)
            break

# Merge files in the same signal group
for i in range(0,len(groupList)):
    inputFileList = " ".join(groupDict[groupList[i]])
    outputFileName = outputLocation+cuellarType+"_"+outputAlias[i]+".bed"
    os.system("cat "+inputFileList+" > "+outputFileName)



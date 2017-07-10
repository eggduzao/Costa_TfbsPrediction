###################################################################################################
# This script should be performed on all Pique results after they are ready.
###################################################################################################

# Import
import os
import sys 
import glob

# Input
signalType = sys.argv[1] # Can be est or def
inputLocation = sys.argv[2]

# Initialization
inputName = "Pique_"
outputLocation = "/".join(inputLocation.split("/")[:-2])+"/"
mergeTempFileName = outputLocation+inputName+signalType+"_merge.bed"
sortFileName = outputLocation+inputName+signalType+".bed"
toDelete = [mergeTempFileName]

###################################################################################################
# 1. Merging all factors
###################################################################################################

# Merging files
os.system("cat "+inputLocation+"*_"+signalType+".bed > "+mergeTempFileName)

# Sorting file
os.system("sort -k1,1 -k2,2n "+mergeTempFileName+" | uniq > "+sortFileName)

# Removing files
for e in toDelete: os.system("rm "+e)



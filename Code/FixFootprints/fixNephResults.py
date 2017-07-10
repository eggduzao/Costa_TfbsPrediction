
# Import
import os
import sys

# Input
inputFileName = sys.argv[1]
outputLocation = sys.argv[2]

# Parameters
ext = "5"
inputName = inputFileName.split("/")[-1].split(".")[0]
outputFileName = outputLocation+inputName+"_"+ext+".bed"

# Merging
os.system("extendAndMerge "+ext+" "+ext+" "+inputFileName+" "+outputFileName)



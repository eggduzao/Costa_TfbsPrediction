#################################################################################################
# Extracts the n top/less entries in a coordinate file by a score corresponding to the file type.
#   .pk files = Gamma score on position 7
#   .bed files = Score on position 4
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. nTop = The number of regions to retrieve. If this number is negative then")
params.append("              it will be returned the n less scored entries.")
params.append("    2. inputFileName = The coordinate file to retrieve the nTop best peaks.")
params.append("    3. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>_(top/less)<nTop>.<fileType> = The retrieved nTop coordinates.")
params.append("###")
#################################################################################################

# Import
import os
import sys
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
nTop = int(sys.argv[1])
inputFileName = sys.argv[2]
outputLocation = sys.argv[3]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Initiating correct score index
scoreIndex = 0
inputFile = open(inputFileName,"r")
sampleLineList = inputFile.readline().strip().split(" ")
if(len(sampleLineList) < 8):
    fileType = "bed"
    scoreIndex = 4
else: 
    fileType = "pk"
    scoreIndex = 7
inputFile.close()

# Extracting each coordinate from file into a dictionary
coordListTemp = aux.createBedListFromSingleFile(inputFileName)

coordList = []
# Removing chrY instances
for v in coordListTemp:
    if(v[0] != "chrY"): coordList.append(v)

# Sorting the lines based on their pValue
coordList = gsort.sortBedList(coordList, field=scoreIndex)
outFileNamePart = "_less"
if(abs(nTop) == nTop): 
    outFileNamePart = "_top"
    coordList = coordList[::-1]

# Writing the nTop lines into output
outputFile = open(outputLocation+inputFileName.split("/")[-1].split(".")[0]+outFileNamePart+str(min(len(coordList),abs(nTop)))+"."+fileType,"w")
for i in range(0,min(len(coordList),abs(nTop))): outputFile.write(" ".join([str(e) for e in coordList[i]])+"\n")

# Termination
outputFile.close()



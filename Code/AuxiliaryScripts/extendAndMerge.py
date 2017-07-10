#################################################################################################
# Extend a group of bed files and then merge them.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. leftExt = Extension to the left of the coordinate file.")
params.append("    2. rightExt = Extension to the right of the coordinate file.")
params.append("    3. coordList = List of coordinate files to be extended and merged.")
params.append("    4. outputFileName = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Result coordinate file.")
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
leftExt = int(sys.argv[1])
rightExt = int(sys.argv[2])
coordList = sys.argv[3].split(",")
outputFileName = sys.argv[4]

# Extend files
outputFile = open(outputFileName+"TEMP","w")
for coordFileName in coordList:
    coordName = coordFileName.split("/")[-1]
    coordFile = open(coordFileName,"r")
    for line in coordFile:
        ll = line.strip().split("\t")
        p1 = max(0,int(ll[1])-leftExt); p2 = int(ll[2])+rightExt
        outputFile.write("\t".join([ll[0],str(p1),str(p2)])+"\n")
    coordFile.close()
outputFile.close()

# Merge coordinates
os.system("sort -k1,1 -k2,2n "+outputFileName+"TEMP | uniq > "+outputFileName+"TEMPS")
os.system("mergeBed -i "+outputFileName+"TEMPS > "+outputFileName)
os.system("rm "+outputFileName+"TEMP "+outputFileName+"TEMPS")



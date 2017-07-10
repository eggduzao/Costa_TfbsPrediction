#################################################################################################
# Crops a certain windowLen of a wig signal file based in bed or pk file coordinates.
# The region cropped corresponds to the expansion of windowLen to the left and right of the
# middle of the intervals in the bed or pk files.
# The length of the cropped fragments will always be 2 * <windowLen>.
# The output is not written sorted by score.
# Ex: windowLength = 4
# Bed Scale:     0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19
# Bed Interval1:                         -------------------- (pos1 = 8, pos2 = 14)
# Region Crop 1:                      ---------------------------
# Bed Interval2:             -------------------- (pos1 = 4, pos2 = 11)
# Region Crop 2:          -----------------------
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. windowLen = Length of the window to be extended for both sides.")
params.append("    2. coordFileName = File containing the coordinates to perform cropping.")
params.append("    3. wigLocation = Folder to the wig files per chromossome.")
params.append("    4. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <wigLocationLastFolder>_mCrop.wig = The resulting cropped file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
import operator
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
windowLen = int(sys.argv[1])
coordFileName = sys.argv[2]
wigLocation = sys.argv[3]
if(wigLocation[-1] != "/"): wigLocation+="/"
outputLocation = sys.argv[4]
if(outputLocation[-1] != "/"): outputLocation+="/"

# File Handling
outputFileNameList = [e for e in wigLocation[:len(wigLocation)-1].split("/") if len(e) >= 2]
if(len(outputFileNameList) >= 2): outputFileName = outputFileNameList[-2]+" - "+outputFileNameList[-1]
else: outputFileName = outputFileNameList[-1]
outputFile = open(outputLocation+outputFileName+"_mCrop.wig","w")

# Verifying file type
isBed = True
tempFile = open(coordFileName,"r")
if(len(tempFile.readline().strip().split(" ")) >= 10): isBed = False
tempFile.close()

# Storing coordinates into disctionary (already in order of score)
coordList = aux.createBedListFromSingleFile(coordFileName)

# Iterating on coordDict to crop desired regions
for coord in coordList:

    # Obtaining real coordenates (rc)
    if(isBed): mPoint =  (coord[1] + coord[2]) / 2
    else: mPoint = coord[1] + coord[9]
    rc = [mPoint - windowLen , mPoint + windowLen]

    # Obtaining bw object
    wigFile = open(glob.glob(wigLocation+coord[0]+"_*.bw")[0],"r")
    bw = BigWigFile(wigFile)

    # Fetching and writing sequence
    bwQuery = bw.get(coord[0],rc[0],rc[1])
    if(bwQuery == None or bwQuery == []): continue
    outputFile.write("fixedStep chrom="+coord[0]+" start="+str(rc[0]+1)+" step=1\n")
    for (c1,c2,value) in bwQuery: outputFile.write(str(round(value,6))+"\n")

    # Closing wig file
    wigFile.close()

# End for chrName in chrNameList

# Termination
outputFile.close()



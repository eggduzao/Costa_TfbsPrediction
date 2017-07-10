#################################################################################################
# Crops a wig file based on coordinates from a bed file.
# Coordinates might not overlap.
# Wig files might be fixedStep only, and sorted by chromossome.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. wigFileName = The file to crop.")
params.append("    2. bedFileName = Coordinates for the cropping.")
params.append("    3. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <wigFileName>_crop.wig = Cropped file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import operator
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
wigFileName = sys.argv[1]
bedFileName = sys.argv[2]
outputLocation = sys.argv[3]
if(outputLocation[-1]!="/"): outputLocation += "/"

# File Handling
outputFileName = outputLocation+wigFileName.split("/")[-1].split(".")[0]+"_crop.wig"
outputFile = open(outputFileName,"w")

# Genome browser header
#fn = wigFileName.split("/")[-1].split(".")[0]
#tn = wigFileName.split("/")[-2]
#outputFile.write("track type=wiggle_0 name="+tn+"_"+fn+" description="+tn+"_"+fn+" visibility=full autoScale=off maxHeightPixels=max")

# Storing coordinates into disctionary and sorting
coordDict = aux.createBedDictFromSingleFile(bedFileName, features = [1, 2], separator="\t")
coordDict = gsort.sortBedDictionaries([coordDict])[0]

# Sorting chrNameList so the output can be in order by chromossome
chrNameList = constants.getChromList(reference=[coordDict])

# Iterating on bed data to fetch the desired regions
for chrName in chrNameList:

    # Obtaining bw object
    wigFile = open(wigFileName,"r")
    bw = BigWigFile(wigFile)

    # Iterating on coordDict to crop desired regions
    for coord in coordDict[chrName]:
        bwQuery = aux.correctBW(bw.get(chrName,coord[0],coord[1]),coord[0],coord[1])
        outputFile.write("fixedStep chrom="+chrName+" start="+str(coord[0]+1)+" step=1\n")
        for value in bwQuery: outputFile.write(str(round(value,6))+"\n")

    # Closing wig file
    wigFile.close()

# End for chrName in chrNameList

# Termination
outputFile.close()

# Converting wig to bigWig and removing wig file
aux.wigToBigWig(outputFileName,outputFileName[:-3]+"bw",removeWig=False)



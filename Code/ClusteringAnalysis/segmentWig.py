#################################################################################################
# Sumarize multiple bw files based on a bed file or do it for whole genome based on chrom
# sizes file.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. anType = Can be 'all' or 'bed'.")
params.append("    2. windowSize = If 'all' then it is the window size to read the genome.")
params.append("    3. normFactorList = List of normalizing factors separated by comma.")
params.append("    4. helpFileName = If 'all' then chrom.sizes. If 'bed' then a coord. file.")
params.append("    5. wigList = List of wig files.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. segmentation.txt = Resulting matrix where each column represents a signal.")
params.append("    2. [wigList].bed = Summary results as bed files.")
params.append("###")
#################################################################################################

# Import
import os
import sys
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
anType = sys.argv[1]
windowSize = sys.argv[2]
if(windowSize != "."): windowSize = int(windowSize)
normFactorList = [float(e) for e in sys.argv[3].split(",")]
helpFileName = sys.argv[4]
wigList = sys.argv[5].split(",")
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"

# If analysis == all
if(anType == "all"):

    # Reading chrom sizes
    chromSizesFile = open(helpFileName,"r")
    chromSizesDict = dict()
    for line in chromSizesFile:
        ll = line.strip().split("\t")
        chromSizesDict[ll[0]] = int(ll[1])

    # Opening signal files
    wigFileList = [open(e,"r") for e in wigList]
    bwFileList = [BigWigFile(e) for e in wigFileList]

    # Iterating on genome
    outputFile = open(outputLocation+"segmentation.txt","w")
    outputFile.write("\t".join([e.split("/")[-1].split(".")[0] for e in wigList])+"\n")
    outputBedFileList = [open(outputLocation+e+".bed","w") for e in [e.split("/")[-1].split(".")[0] for e in wigList]]
    counter = 0
    for chrName in constants.getChromList():
        if(chrName not in chromSizesDict.keys()): continue
        for p1 in xrange(0,chromSizesDict[chrName],windowSize):
            p2 = min(p1+windowSize,chromSizesDict[chrName])
            counter += 1
            toWrite = []
            for i in range(0,len(bwFileList)):
                sequence = aux.correctBW(bwFileList[i].get(chrName,p1,p2),p1,p2)
                value = str((float(sum(sequence))*normFactorList[i])/len(sequence))
                toWrite.append(value)
                outputBedFileList[i].write("\t".join([chrName,str(p1),str(p2),"s"+str(counter),value,"."])+"\n")
            outputFile.write("\t".join(toWrite)+"\n")

    # Termination
    chromSizesFile.close()
    outputFile.close()
    for e in wigFileList: e.close()
    for e in outputBedFileList: e.close()

# If analysis == bed
else:

    # Opening signal files
    wigFileList = [open(e,"r") for e in wigList]
    bwFileList = [BigWigFile(e) for e in wigFileList]

    # Iterating on bed file
    coordFile = open(helpFileName,"r")
    outputFile = open(outputLocation+"segmentation.txt","w")
    outputFile.write("\t".join([e.split("/")[-1].split(".")[0] for e in wigList])+"\n")
    outputBedFileList = [open(outputLocation+e+".bed","w") for e in [e.split("/")[-1].split(".")[0] for e in wigList]]
    counter = 0
    for line in coordFile:

        ll = line.strip().split("\t")
        chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])
        counter += 1

        # Iterating on bw signals
        toWrite = []
        for i in range(0,len(bwFileList)):
            sequence = aux.correctBW(bwFileList[i].get(chrName,p1,p2),p1,p2)
            value = str((float(sum(sequence))*normFactorList[i])/len(sequence))
            toWrite.append(value)
            outputBedFileList[i].write("\t".join([chrName,str(p1),str(p2),"s"+str(counter),value,"."])+"\n")
        outputFile.write("\t".join(toWrite)+"\n")

    # Termination
    coordFile.close()
    outputFile.close()
    for e in wigFileList: e.close()



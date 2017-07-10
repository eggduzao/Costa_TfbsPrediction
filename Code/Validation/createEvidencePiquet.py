#################################################################################################
# Creates a bed file with MPBSs with (in green) and without (in red) evidence.
# Also, the name of the instances will be Y for evidence, N for non evidence and . for rest.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. peakExt = Extension of the peaks summit (total window length).")
params.append("    2. negExt = Extension to count signals to create negative set (total).")
params.append("    3. mpbsName = Name of the MPBS.")
params.append("    4. tfbsSummitFileName = Location + name of the TFBS ChIP-seq summit file.")
params.append("    5. treatFileName = Location + name of the ChIP-seq treatment signal.")
params.append("    6. controlFileName = Location + name of the ChIP-seq control signal.")
params.append("    7. mpbsFileName = File containing MBPSs.")
params.append("    8. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <mpbsName>.bed = Result evidence bed file.")
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
peakExt = int(sys.argv[1])
negExt = int(sys.argv[2])
mpbsName = sys.argv[3]
tfbsSummitFileName = sys.argv[4]
treatFileName = sys.argv[5]
controlFileName = sys.argv[6]
mpbsFileName = sys.argv[7]
outputLocation = sys.argv[8]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Reading MPBS
mpbsDictTemp = aux.createBedDictFromSingleFile(mpbsFileName, separator="\t")
mpbsDict = dict()
for k in mpbsDictTemp.keys():
    for e in mpbsDictTemp[k]:
        if(e[2] == mpbsName):
            if(k in mpbsDict): mpbsDict[k].append(e)
            else: mpbsDict[k] = [e]
mpbsDict = gsort.sortBedDictionaries([mpbsDict])[0]

# Reading TFBS
tfbsSummitFile = open(tfbsSummitFileName,"r")
tfbsDict = dict([(e,[]) for e in mpbsDict.keys()])
for sVec in tfbsSummitFile:
    sVec = sVec.strip().split("\t")
    if(sVec[0] not in mpbsDict.keys()): continue
    sVec[1] = max(int(sVec[1])-(peakExt/2),0)
    sVec[2] = int(sVec[2])+(peakExt/2)
    tfbsDict[sVec[0]].append(sVec[1:])
tfbsSummitFile.close()
tfbsDict = gsort.sortBedDictionaries([tfbsDict])[0]

# Reading wig files
treatFile = open(treatFileName,"r")
bwTreat = BigWigFile(treatFile)
controlFile = open(controlFileName,"r")
bwControl = BigWigFile(controlFile)

# Evaluating overlap
for chrom in constants.getChromList(reference=[mpbsDict]):
    c = 0
    for interval in mpbsDict[chrom]:
        didBreak = False
        while(c < len(tfbsDict[chrom])):
            check = aux.checkTuplesOverlap(interval,tfbsDict[chrom][c])
            if(check == 0):
                interval[2] = mpbsName+":Y"
                interval += [interval[0],interval[1],"0,150,0"]
                didBreak = True
                break
            elif(check == -1):
                treatSignalQ = bwTreat.get(chrom,max(interval[0]+(peakExt/2)-(negExt/2),0),interval[1]-(peakExt/2)+(negExt/2))
                controlSignalQ = bwControl.get(chrom,max(interval[0]+(peakExt/2)-(negExt/2),0),interval[1]-(peakExt/2)+(negExt/2))
                treatSum = sum([e[2] for e in treatSignalQ]); controlSum = sum([e[2] for e in controlSignalQ])
                if(treatSum > controlSum):
                    interval[2] = mpbsName+":."
                    interval += [interval[0],interval[1],"0,0,0"]
                else:
                    interval[2] = mpbsName+":N"
                    interval += [interval[0],interval[1],"150,0,0"]
                didBreak = True
                break
            c += 1
        if(not didBreak):
            treatSignalQ = bwTreat.get(chrom,max(interval[0]+(peakExt/2)-(negExt/2),0),interval[1]-(peakExt/2)+(negExt/2))
            controlSignalQ = bwControl.get(chrom,max(interval[0]+(peakExt/2)-(negExt/2),0),interval[1]-(peakExt/2)+(negExt/2))
            treatSum = sum([e[2] for e in treatSignalQ]); controlSum = sum([e[2] for e in controlSignalQ])
            if(treatSum > controlSum):
                interval[2] = mpbsName+":."
                interval += [interval[0],interval[1],"0,0,0"]
            else:
                interval[2] = mpbsName+":N"
                interval += [interval[0],interval[1],"150,0,0"]

# Writing
outputFile = open(outputLocation+mpbsName+".bed","w")
for chrom in constants.getChromList(reference=[mpbsDict]):
    for e in mpbsDict[chrom]: outputFile.write("\t".join([str(k) for k in [chrom]+e])+"\n")

# Termination
treatFile.close()
controlFile.close()
outputFile.close()



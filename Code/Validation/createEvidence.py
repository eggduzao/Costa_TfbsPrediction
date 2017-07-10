#################################################################################################
# Creates a bed file with MPBSs with (in green) and without (in red) evidence.
# Also, the name of the instances will be Y for evidence, N for non evidence.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. peakExt = Extension of the peaks summit (total window length).")
params.append("    2. mpbsName = Name of the MPBS.")
params.append("    3. tfbsSummitFileName = Location + name of the TFBS ChIP-seq summit file.")
params.append("    4. mpbsFileName = File containing MBPSs.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputName>.bed = Result evidence bed file.")
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
peakExt = int(sys.argv[1])
mpbsName = sys.argv[2]
tfbsSummitFileName = sys.argv[3]
mpbsFileName = sys.argv[4]
outputLocation = sys.argv[5]
if(outputLocation[-1]!="/"): outputLocation += "/"

# Initializing outputName
# It was previously the MPBS name, but now I changed to the TFBS, since there are equal motifs for different ChIP experiments.
outputName = tfbsSummitFileName.split("/")[-1].split(".")[0] # Name to be written is ChIP-seq TFBS name
#outputName = mpbsName # Name to be written is motif matching MPBS name

# Separating factor
mpbsFile = open(mpbsFileName,"r")
mpbsSepName = outputLocation+outputName+"_sep.bed"
mpbsOutFile = open(mpbsSepName,"w")
for line in mpbsFile:
    if(mpbsName in line): mpbsOutFile.write(line)
mpbsFile.close()
mpbsOutFile.close()

# Expanding summits
tfbsSummitFile = open(tfbsSummitFileName,"r")
tfbsExtName = outputLocation+outputName+"_tfbs.bed"
tfbsExtFile = open(tfbsExtName,"w")
for line in tfbsSummitFile:
    ll = line.strip().split("\t")
    mid = (int(ll[1])+int(ll[2]))/2
    p1 = max(mid-(peakExt/2),0); p2 = mid+(peakExt/2)
    tfbsExtFile.write("\t".join([ll[0]]+[str(p1),str(p2)]+ll[3:])+"\n")
tfbsSummitFile.close()
tfbsExtFile.close()

# Calculating intersections
intFileName = outputLocation+outputName+"_int.bed"
nintFileName = outputLocation+outputName+"_nint.bed"
mpbsSepSortName = outputLocation+outputName+"_sep_sort.bed"
tfbsExtSortName = outputLocation+outputName+"_tfbs_sort.bed"
os.system("sort -k1,1 -k2,2n "+mpbsSepName+" > "+mpbsSepSortName)
os.system("sort -k1,1 -k2,2n "+tfbsExtName+" > "+tfbsExtSortName)
os.system("intersectBed -a "+mpbsSepSortName+" -b "+tfbsExtSortName+" -wa > "+intFileName)
os.system("intersectBed -a "+mpbsSepSortName+" -b "+tfbsExtSortName+" -wa -v > "+nintFileName)

# Writing results
outputUnsortFileName = outputLocation+outputName+"_unsort.bed"
outputFile = open(outputUnsortFileName,"w")
intFile = open(intFileName,"r")
for line in intFile:
    ll = line.strip().split("\t")
    outputFile.write("\t".join(ll[0:3]+[outputName+":Y",ll[4],ll[5],ll[1],ll[2],"0,150,0"])+"\n")
intFile.close()
nintFile = open(nintFileName,"r")
for line in nintFile:
    ll = line.strip().split("\t")
    outputFile.write("\t".join(ll[0:3]+[outputName+":N",ll[4],ll[5],ll[1],ll[2],"150,0,0"])+"\n")
nintFile.close()
outputFile.close()

# Sorting file
outputFileName = outputLocation+outputName+".bed"
os.system("sort -k1,1 -k2,2n "+outputUnsortFileName+" > "+outputFileName)

# Removing files
os.system("rm "+mpbsSepName+" "+tfbsExtName+" "+intFileName+" "+nintFileName+" "+mpbsSepSortName+" "+tfbsExtSortName+" "+outputUnsortFileName)



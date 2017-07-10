#################################################################################################
# Runs Fseq program keeping the necessary files (counts) and discarding log files.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. fragSize = Size of the reads. For DNase use 0, for Histone use 200.")
params.append("    2. featureLength = Feature length. For DNase use 600, for Histone use 300.")
params.append("    3. backLocation = Location of the background (bff) files.")
params.append("    4. ploidyLocation = Location of the ploidy (iff) files.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("    6. [folderList] = List of folders containing .bed files (one for each chrom.")
params.append("       i.e. output of sepRepByChr.py). Each folder represents one cell line.")
params.append("###")
params.append("Output: ")
params.append("    1. [<chrName>_counts.wig] = List of output WIG files (per chromossome).")
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
fragSize = int(sys.argv[1])
featureLength = int(sys.argv[2])
backLocation = sys.argv[3]
if(backLocation[-1] != "/"): backLocation+="/"
ploidyLocation = sys.argv[4]
if(ploidyLocation[-1] != "/"): ploidyLocation+="/"
outputLocation = sys.argv[5]
if(outputLocation[-1] != "/"): outputLocation+="/"
folderList = sys.argv[6:]
for i in range(0,len(folderList)):
    if(folderList[i][-1] != "/"): folderList[i]+="/"

# Merging all input files from all input folders
mergeCmd = "bedMerge "+outputLocation+"merged.bed"
for e in folderList: mergeCmd += " "+e+"*"
os.system(mergeCmd)

# Running Fseq
fseqCmd = "fseq"
fseqCmd += " -d "+outputLocation
fseqCmd += " -o "+outputLocation
fseqCmd += " -b "+backLocation
fseqCmd += " -p "+ploidyLocation
fseqCmd += " -of wig"
fseqCmd += " -f "+str(fragSize)
fseqCmd += " -l "+str(featureLength)
fseqCmd += " -s 1"
fseqCmd += " merged.bed"
fseqCmd += " > "+outputLocation+"fseqLog.txt"
os.system(fseqCmd)

# Deleting possibly old files
for fileName in os.listdir(outputLocation):
    if("_counts" in fileName): os.system("rm "+outputLocation+fileName)

# Renaming Fseq output to correct name (with _count)
for fileName in os.listdir(outputLocation):
    if(".wig" in fileName):
        chrName = fileName.split(".")[0]
        outputFileName = outputLocation+chrName+"_counts.wig"
        os.system("mv "+outputLocation+chrName+".wig "+outputFileName)
        aux.wigToBigWig(outputFileName,outputFileName[:-3]+"bw",removeWig=False)

# Termination - Removing unecessary files
os.system("rm "+outputLocation+"merged.bed")
os.system("rm "+outputLocation+"fseqLog.txt")



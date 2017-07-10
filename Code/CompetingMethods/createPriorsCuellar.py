#################################################################################################
# Script to create the priors used by FIMO in Cuellar method.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. alphaValue = Cuellar parameter.")
params.append("    2. betaValue = Cuellar parameter.")
params.append("    3. chromSizesFileName = Name of the chrom.sizes file used.")
params.append("    4. inputFileName = Name of the input file.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. [chr*.prior] = Prior values separated by chromosome.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
alphaValue = sys.argv[1]
betaValue = sys.argv[2]
chromSizesFileName = sys.argv[3]
inputFileName = sys.argv[4]
outputLocation = sys.argv[5]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Create output folder
outputName = inputFileName.split("/")[-1].split(".")[0]
outputFolder = outputLocation+outputName+"/"
os.system("mkdir -p "+outputFolder)

# Read chrom sizes
chromSizesFile = open(chromSizesFileName,"r")
chromSizesDict = dict()
for line in chromSizesFile:
    ll = line.strip().split("\t")
    chromSizesDict[ll[0]] = ll[1]
chromSizesFile.close()

# Converting input bw to wig
newInputFileName = outputFolder+outputName+".wig"
os.system("bigWigToWig "+inputFileName+" "+newInputFileName)

# Separate by chromosome
currChr = ""
currOutputFile = None
newInputFile = open(newInputFileName,"r")
for line in newInputFile:
    if(line[0] == "f"):
        chrName = line.strip().split(" ")[1][6:]
        if(chrName != currChr):
            currChr = chrName
            if(currOutputFile): currOutputFile.close()
            currOutputFile = open(outputFolder+chrName+".wig","w")
    currOutputFile.write(line)
newInputFile.close()
currOutputFile.close()

# Evaluating chromosome list
chrList = [e.split("/")[-1].split(".")[0] for e in glob.glob(outputFolder+"chr*.wig")]
    
# Iterating on chromosome list
for chrName in chrList:

    # Calculating priors
    os.system("createprior -a "+alphaValue+" -b "+betaValue+" -i "+outputFolder+chrName+".wig -o "+outputFolder+"priors"+" -n "+chromSizesDict[chrName])

    # Fixing priors (removing multiple headers)
    priorFile = open(outputFolder+"priors/"+chrName+".prior","r")
    headerLine = priorFile.readline()
    priorFile.close()
    os.system("grep -v variable "+outputFolder+"priors/"+chrName+".prior > "+outputFolder+"priors/"+chrName+".priortemp")
    os.system("sed -i -e '1i"+headerLine+"' "+outputFolder+"priors/"+chrName+".priortemp")

    # Fixing folder
    os.system("mv "+outputFolder+"priors/"+chrName+".priortemp "+outputFolder+chrName+".prior")
    os.system("rm -rf "+outputFolder+"priors/")
    os.system("rm "+outputFolder+chrName+".wig")

# Termination
os.system("rm "+newInputFileName)



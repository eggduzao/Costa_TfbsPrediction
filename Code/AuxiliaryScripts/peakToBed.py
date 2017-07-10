#################################################################################################
# Converts a narrowPeak or BroadPeak (PK) file to BED format
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. separator = Field separator of the .taf file. Can be \"tab\" or \"spc\".")
params.append("    2. inputFileName = The pk file.")
params.append("    3. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>.bed = The converted bed files.")
params.append("###")
#################################################################################################

# Import
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
separator = sys.argv[1]
if(separator == "tab"): separator = "\t"
else: separator = " "
inputFileName = sys.argv[2]
outputLocation = sys.argv[3]
if(outputLocation[-1]!="/"): outputLocation += "/"

# File handling
inputFile = open(inputFileName,"r")
outputFile = open(outputLocation+inputFileName.split("/")[-1].split(".")[0]+".bed","w")
    
# Write output
for line in inputFile:
    if(len(line) < 2): continue
    line = line.strip()
    lineList = line.split(separator)
    outputFile.write(lineList[0]+"\t"+lineList[1]+"\t"+lineList[2]+"\t"+lineList[3]+"\t"+lineList[4]+"\t"+lineList[5]+"\n")

# Termination
inputFile.close()
outputFile.close()



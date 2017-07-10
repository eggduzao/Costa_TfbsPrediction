#################################################################################################
# Converts a .taf file into a .bed file
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. threshold = Cutoff threshold. Must be an integer within [0,1000].")
params.append("                 This may be used, for example, to cut off badly aligned reads.")
params.append("    2. separator = Field separator of the .taf file. Can be \"tab\" or \"spc\".")
params.append("    3. inputFileName = The taf file with the following fields:")
params.append("                 <chromossome> <pos1> <pos2> <sequence> <score> <strand>")
params.append("    4. outputFileName = The location + name of the outputFile.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>.bed = The corresponding bed file.")
params.append("###")
#################################################################################################

# Import
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
threshold = int(sys.argv[1])
separator = sys.argv[2]
if(separator=="tab"): separator = "\t"
else: separator = " "
inputFileName = sys.argv[3]
outputFileName = sys.argv[4]

# File handling
inputFile = open(inputFileName,"r")
outputFile = open(outputFileName,"w")

# Iterate on the taf file
counter = 0
for line in inputFile:

    if(len(line) < 2): continue
    
    # Writes the line into the .bed output
    lineList = line.strip().split(separator)
    if(int(lineList[4]) > threshold):
        outputFile.write(lineList[0]+"\t"+lineList[1]+"\t"+lineList[2]+"\tr"+str(counter)+"\t"+lineList[4]+"\t"+lineList[5]+"\n")
        counter+=1

# End for line in inputFile

# Termintion
outputFile.close()
inputFile.close()


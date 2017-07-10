#################################################################################################
# Applies Neph footprinting algorithm and outputs a bed file with footprints.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. fosThresh = Threshold for significance cutoff. Eg. 0.95.")
params.append("    2. coordFileName = Coordinates in which to apply Neph.")
params.append("    3. dnaseFileName = Bigwig file containing DNase counts.")
params.append("    4. outputFileName = Name of the output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputfileName>.bed")
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
fosThresh = float(sys.argv[1])
coordFileName = sys.argv[2]
dnaseFileName = sys.argv[3]
outputFileName = sys.argv[4]

# File parameters
outputLocation = "/".join(outputFileName.split("/")[:-1])+"/"
nephSignalFileName = outputLocation+"nephsignal.txt"
nephResultFileName = outputLocation+"nephresult.txt"
dnaseFile = open(dnaseFileName,"r")
bw = BigWigFile(dnaseFile)

# Iterating on coordinate file
coordFile = open(coordFileName,"r")
outputFile = open(outputFileName,"w")
counter = 1
for line in coordFile:

    # Initialization
    ll = line.strip().split("\t")
    chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])

    # Create Neph signal input
    nephSignalFile = open(nephSignalFileName,"w")
    nephSignalFile.write("\n".join([str(e) for e in aux.correctBW(bw.get(chrName,p1,p2),p1,p2)]))
    nephSignalFile.close()

    # Apply neph
    nephCommand = "detect-cache "
    nephCommand += "--flankmin 3 --flankmax 10 --centermin 6 --centermax 40 --maxthold 10 "
    nephCommand += nephSignalFileName+" > "+nephResultFileName
    os.system(nephCommand)
    os.system("rm "+nephSignalFileName)

    # Read/Write neph results
    nephResultFile = open(nephResultFileName,"r")
    for line in nephResultFile:
        ll = line.strip().split("\t")
        if(float(ll[4]) >fosThresh): continue
        outputFile.write("\t".join([chrName,str(p1+int(ll[1])),str(p1+int(ll[2])),"n"+str(counter),ll[4],"."])+"\n")
        counter += 1
    nephResultFile.close()
    os.system("rm "+nephResultFileName)

dnaseFile.close()
coordFile.close()
outputFile.close()



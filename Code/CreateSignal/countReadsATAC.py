#################################################################################################
# Counts the reads of ATAC-seq experiments with the correct read shifting.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. extParam = 4,-5,0,1 for ATAC and 0,0,0,9 for ATAC_BO.")
params.append("    2. bamFileName = Bam file in which to create count signal.")
params.append("    3. csFileName = Chromosome sizes file name.")
params.append("    4. outputFileName = Name of output wig file.")
params.append("###")
params.append("Output: ")
params.append("    1. outputFileName = Resulting wig file.")
params.append("###")
#################################################################################################

#################################################################################################
# INPUT
#################################################################################################

# Import
import os
import sys
from pysam import Samfile
from multiprocessing import Pool
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
extParam = sys.argv[1].split(",")
bamFileName = sys.argv[2]
csFileName = sys.argv[3]
outputFileName = sys.argv[4]

# Parameters
fShift = int(extParam[0])
rShift = int(extParam[1])
leftExt = int(extParam[2])
rightExt = int(extParam[3])
nproc = 10
regionLen = 100000

#################################################################################################
# FUNCTIONS
#################################################################################################

# Creating bam file
bamFile = Samfile(bamFileName,"rb")

# PileupRegion Class
class PileupRegion:
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.length = end-start
        self.vector = [0.0] * self.length

    def __call__(self, alignment):
        if(not alignment.is_reverse):
            p1 = max(alignment.pos+fShift-leftExt,self.start)-self.start
            p2 = min(p1 + rightExt, self.length)
        else:
            p2 = min(alignment.aend+rShift+leftExt,self.end)-self.start
            p1 = max(p2 - rightExt, 0)
        for i in range(p1, p2): self.vector[i] += 1.0

# Function to read pileup
def read_pileup((chrName, start, end)):
    pileup_region = PileupRegion(start,end)
    iter = bamFile.fetch(reference=chrName, start=start, end=end)
    for alignment in iter: pileup_region.__call__(alignment)
    #bamFile.fetch(reference=chrName, start=start, end=end, callback = pileup_region)
    return pileup_region.vector

#################################################################################################
# Execution
#################################################################################################

# Initialization
outputFile = open(outputFileName,"w")

# Reading csDict
csDict = dict()
csFile = open(csFileName,"r")
for line in csFile:
    ll = line.strip().split("\t")
    csDict[ll[0]] = int(ll[1])
csFile.close()
chrList = sorted(csDict.keys())

# Creating input
inputList = []
for chrName in chrList:
    for i in range(0,csDict[chrName],regionLen): inputList.append((chrName,i,min(i+regionLen,csDict[chrName])))

# Grouping input by the number of processes requested
if(nproc == 1): inputList = [[e] for e in inputList]
else: inputList = map(None, *(iter(inputList),) * nproc)

# Iterating on input groups
for inputGroup in inputList:

    # Removing Nones
    inputGroup = [e for e in inputGroup if e]

    # Applying pileup function
    curr_proc_nb = len(inputGroup)
    pool = Pool(curr_proc_nb)
    resultList = map(read_pileup,inputGroup)
    pool.close()
    pool.join()

    # Writing results to wig file
    for i in range(0,len(inputGroup)):
        outputFile.write("fixedStep chrom="+inputGroup[i][0]+" start="+str(inputGroup[i][1]+1)+" step=1\n")
        for v in resultList[i]: outputFile.write(str(v)+"\n")

# Termination
bamFile.close()
outputFile.close()

# Converting to bigwig
os.system(" ".join(["wigToBigWig",outputFileName,csFileName,outputFileName[:-3]+"bw"]))
#os.system(" ".join(["rm",outputFileName]))



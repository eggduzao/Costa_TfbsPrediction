#################################################################################################
# Counts the reads for each bp for each chromossome and saves this information into a wig 
# (fixedStep) file with a bp resolution.
### USAGE:
# DNase counts: leftInc = 0, rightInc = 1          -> Reads only the 5' bp.
# DNase base overlap: leftInc = -5, rightInc = 5   -> Extends the read 5bp to the left and right.
# Histone counts: leftInc = 0, rightInc = 200      -> Reads the first 200bp from the 5' bp.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. leftInc = Amount of bases to increase from the 5' bp.")
params.append("    2. rightInc = Amount of bases to increase from the 5' bp.")
params.append("    3. bamFileName = Bam file in which to create count signal.")
params.append("    4. csFileName = Chromosome sizes file name.")
params.append("    5. outputFileName = Name of output wig file.")
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
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
leftInc = int(sys.argv[1])
rightInc = int(sys.argv[2])
bamFileName = sys.argv[3]
csFileName = sys.argv[4]
outputFileName = sys.argv[5]

# Parameters
nproc = 10
regionLen = 100000

#################################################################################################
# FUNCTIONS
#################################################################################################

# Creating bam file
bamFile = Samfile(bamFileName,"rb")

# PileupRegion Class
class PileupRegion:
    def __init__(self,start,end,lext,rext):
        self.start = start
        self.end = end
        self.length = end-start
        self.lext = lext
        self.rext = rext
        self.vector = [0.0] * self.length

    def __call__(self, alignment):
        if(not alignment.is_reverse):
            for i in range(max(alignment.pos-self.lext,self.start),min(alignment.pos+self.rext,self.end-1)):
                self.vector[i-self.start] += 1.0
        else:
            for i in range(max(alignment.aend-self.rext,self.start),min(alignment.aend+self.lext,self.end-1)):
                self.vector[i-self.start] += 1.0 

# Function to read pileup
def read_pileup((chrName, start, end)):
    pileup_region = PileupRegion(start,end,leftInc,rightInc)
    #bamFile.fetch(reference=chrName, start=start, end=end, callback = pileup_region)
    iter = bamFile.fetch(reference=chrName, start=start, end=end)
    for alignment in iter: pileup_region.__call__(alignment)
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



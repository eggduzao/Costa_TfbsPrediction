
# Import 
import os
import sys
from pysam import Samfile

# Input
lab = sys.argv[1]
cell = sys.argv[2]

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
def read_pileup(bamFile, chrName, start, end):
    pileup_region = PileupRegion(start,end,0,1)
    iter = bamFile.fetch(reference=chrName, start=start, end=end)
    for alignment in iter: pileup_region.__call__(alignment)
    return pileup_region.vector

clipN = 100.0

# Parameters
bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNase.bam"
hsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNasePeaks.bed"
tgFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/SpecialCoverage/frip/tiled_genome.bed"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/SpecialCoverage/frip/"+lab+"_"+cell+".txt"

# Evaluating HS coverage
bamFile = Samfile(bamFileName,"rb")
inFile = open(hsFileName,"r")
hsN = 0.0
for line in inFile:
  ll = line.strip().split("\t")
  hsN += sum([min(e,clipN) for e in read_pileup(bamFile, ll[0], int(ll[1]), int(ll[2]))])
inFile.close()

# Evaluating total coverage
inFile = open(tgFileName,"r")
hsT = 0.0
for line in inFile:
  ll = line.strip().split("\t")
  hsT += sum([min(e,clipN) for e in read_pileup(bamFile, ll[0], int(ll[1]), int(ll[2]))])
inFile.close()

# Writing results
outFile = open(outFileName,"w")
outFile.write("\t".join([str(e) for e in [hsN,hsT,float(hsN)/hsT]]))

# Termination
bamFile.close()
outFile.close()



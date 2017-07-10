#################################################################################################
# Create DNase cuts and fasta file for FootprintMixture sogtware [Ohler et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsFileName = BED file with the MPBSs.")
params.append("    2. bamFileName = BAM file with the reads.")
params.append("    3. genomeFileName = FASTA file with genome.")
params.append("    4. outputFileName = Location + prefix of output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputfileName>_CUT.txt = Text file containing the cuts.")
params.append("    2. <outputfileName>_SEQ.fa = Text file containing the sequences.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import pysam as ps
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsFileName = sys.argv[1]
bamFileName = sys.argv[2]
genomeFileName = sys.argv[3]
outputFileName = sys.argv[4]

# Pileup class
class PileupRegion:
  def __init__(self,start,end,ext):
    self.start = start
    self.end = end
    self.length = end-start
    self.ext = ext
    self.vector = [0.0] * self.length
  def __call__(self, alignment):
    if(not alignment.is_reverse):
      for i in range(max(alignment.pos,self.start),min(alignment.pos+self.ext,self.end-1)):
        self.vector[i-self.start] += 1.0
    else:
      if(alignment.aend):
        for i in range(max(alignment.aend-self.ext,self.start),min(alignment.aend,self.end-1)):
          self.vector[i-self.start] += 1.0

# Creating cuts file
ext = 25
mpbsFile = open(mpbsFileName,"r")
bamFile = ps.AlignmentFile(bamFileName, "rb")
fastaFile = ps.FastaFile(genomeFileName)
cutFile = open(outputFileName+"_CUT.txt","w")
seqFile = open(outputFileName+"_SEQ.fa","w")
counter = 1
for line in mpbsFile:
  ll = line.strip().split("\t")
  if(int(ll[1])-ext-3 <= 0): continue
  pileup_region = PileupRegion(int(ll[1])-ext,int(ll[2])+ext,1)
  iter = bamFile.fetch(reference=ll[0], start=int(ll[1])-ext, end=int(ll[2])+ext) 
  for alignment in iter: pileup_region.__call__(alignment) 
  currseq = fastaFile.fetch(reference=ll[0], start=int(ll[1])-ext-3, end=int(ll[2])+ext+3)
  cutFile.write(" ".join([str(e) for e in pileup_region.vector])+"\n")
  seqFile.write(">"+str(counter)+"\n")
  seqFile.write(currseq.upper()+"\n")
  counter += 1
bamFile.close()
mpbsFile.close()




# Import
import os
import sys
from pysam import Samfile

class PileupRegion:
  def __init__(self,start,end,ext):
    self.start = start
    self.end = end
    self.length = end-start
    self.ext = ext
    self.vectorF = [0.0] * self.length
    self.vectorR = [0.0] * self.length
  def __call__(self, alignment):
    if(not alignment.is_reverse):
      for i in range(max(alignment.pos,self.start),min(alignment.pos+self.ext,self.end-1)):
        self.vectorR[i-self.start] += 1.0 
    else:
      for i in range(max(alignment.aend-self.ext,self.start),min(alignment.aend,self.end-1)):
        self.vectorF[i-self.start] += 1.0 

# Input
bamFileName = sys.argv[1]
hsFileName = sys.argv[2]
outputfileName = sys.argv[3]
initial_clip = 1000
ext = 1

# Opening files
bamFile = Samfile(bamFileName,"rb")
hsFile = open(hsFileName,"r")

# Resulting statistics
O_Plus = 0.0
O_Minus = 0.0
R = 0.0

# Iterating on HS regions
for line in hsFile:

  # Fetching signal
  ll = line.strip().split("\t")
  pileup_region = PileupRegion(int(ll[1]),int(ll[2]),ext)
  iter = bamFile.fetch(reference=ll[0], start=int(ll[1]), end=int(ll[2]))
  for alignment in iter: pileup_region.__call__(alignment)
  raw_signalF = [min(e,initial_clip) for e in pileup_region.vectorF]
  raw_signalR = [min(e,initial_clip) for e in pileup_region.vectorR]

  # Updating statistics
  O_Plus += sum(raw_signalF)
  O_Minus += sum(raw_signalR)
  R += int(ll[2])-int(ll[1])

# Writing results
outputFile = open(outputfileName,"w")
outputFile.write("\t".join(["O+","O-","R","NormFactor+","NormFactor-"])+"\n")
outputFile.write("\t".join([str(O_Plus),str(O_Minus),str(R),str(R/O_Plus),str(R/O_Minus)])+"\n")

# Termination
bamFile.close()
hsFile.close()
outputFile.close()



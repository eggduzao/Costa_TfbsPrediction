
###########################################################
# INPUT
###########################################################

# Import
import os
import sys
from math import log
from pysam import Samfile
from numpy import exp, array, abs, int, mat, linalg, convolve

# Input
totalWindow = int(sys.argv[1])
bedFileNameList = sys.argv[2].split(",")
bamFileName = sys.argv[3]
outFileName = sys.argv[4]

signalExt = int(sys.argv[5])
signalExtBoth = sys.argv[6]
if(signalExtBoth == "Y"): signalExtBoth = True
else: signalExtBoth = False

###########################################################
# CLASSES
###########################################################

# Class PileupRegion
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
            for i in range(max(alignment.aend-self.ext,self.start),min(alignment.aend,self.end-1)):
                self.vector[i-self.start] += 1.0 
    def __call2__(self, alignment):
        if(not alignment.is_reverse):
            for i in range(max(alignment.pos-self.ext,self.start),min(alignment.pos+self.ext,self.end-1)):
                self.vector[i-self.start] += 1.0 
        else:
            for i in range(max(alignment.aend-self.ext,self.start),min(alignment.aend+self.ext,self.end-1)):
                self.vector[i-self.start] += 1.0

class GenomicSignal:
    def __init__(self, file_name):
        self.bam = Samfile(file_name,"rb")
    def get_signal(self, ref, start, end, ext, initial_clip = 1000, ext_both_directions=False):
        pileup_region = PileupRegion(start,end,ext)
        iter = self.bam.fetch(reference=ref, start=start, end=end)
        if(not ext_both_directions):
            for alignment in iter: pileup_region.__call__(alignment)
        else:
            for alignment in iter: pileup_region.__call2__(alignment)
        raw_signal = array([min(e,initial_clip) for e in pileup_region.vector])
        mean = raw_signal.mean()
        std = raw_signal.std()
        clip_signal = [min(e, mean + (10 * std)) for e in raw_signal]

        return clip_signal

###########################################################
# EXECUTION
###########################################################

# Initialization
resList = []
bamFile = GenomicSignal(bamFileName)

# Iterating on bed file list
for bedFileName in bedFileNameList:

  # Iterating on bed files
  res = [0.0] * totalWindow
  total = 0
  bedFile = open(bedFileName,"r")
  for line in bedFile:
    ll = line.strip().split("\t")
    mid = (int(ll[1]) + int(ll[2])) / 2
    p1 = mid - (totalWindow/2); p2 = mid + (totalWindow/2) + 1
    try: signal = bamFile.get_signal(ll[0], p1, p2, signalExt, initial_clip = 1000, ext_both_directions=signalExtBoth)
    except Exception: continue
    for i in range(0,len(signal)-1): res[i] += signal[i]
    total += 1
  bedFile.close()
  for i in range(0,len(res)): res[i] /= float(total)
  resList.append(res)

###########################################################
# WRITING
###########################################################

outFile = open(outFileName,"w")
outFile.write("\t".join([e.split("/")[-1].split(".")[0] for e in bedFileNameList])+"\n")
for i in range(0,len(resList[0])):
  vec = []
  for j in range(0,len(resList)): vec.append(str(resList[j][i]))
  outFile.write("\t".join(vec)+"\n")
outFile.close()



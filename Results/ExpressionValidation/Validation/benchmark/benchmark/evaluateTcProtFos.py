
# Import
import os
import sys
from glob import glob
from bx.bbi.bigwig_file import BigWigFile
sys.path.append("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code")
from util import *

# Reading input
mpbsFileName = sys.argv[1]
signalFileName = sys.argv[2]
outputFileNameFos = sys.argv[3]
halfWindow = 100

# File handling
signalFile = open(signalFileName,"r")
bw = BigWigFile(signalFile)
mpbsFile = open(mpbsFileName,"r")
outputFileFos = open(outputFileNameFos,"w")

# Minvalue
minV = -4
maxV = 4

# Fetching coverage
covDict = dict()
covList = glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/SpecialCoverage/special_coverage/*.txt")
for inFileName in covList:
  inName = inFileName.split("/")[-1].split(".")[0]
  inFile = open(inFileName,"r")
  covDict[inName] = 1000000.0/float(inFile.readline().strip())
  inFile.close()
cov = covDict[signalFileName.split("/")[-2]]

# Iterating on the mpbsfile to update the score
for line in mpbsFile:

  # Initialization
  ll = line.strip().split()
  chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])

  # FS
  mLen = p2-p1
  p1_ext = p1 - mLen
  p2_ext = p2 + mLen
  if(p1_ext < 0): continue
  try: signalVec = [ (min(max(e,minV),maxV)-minV)*cov for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]
  except Exception: continue
  nl = sum(signalVec[:mLen]) + 1.0
  nc = sum(signalVec[mLen:2*mLen]) + 1.0
  nr = sum(signalVec[2*mLen:]) + 1.0
  try: nFos = str(round(-((nc/nr)+(nc/nl)),4))
  except Exception: continue

  # Writing
  outputFileFos.write("\t".join(ll[:4]+[nFos])+"\n")

# Termination
signalFile.close()
mpbsFile.close()
outputFileFos.close()



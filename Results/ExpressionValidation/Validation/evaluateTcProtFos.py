
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
outputFileNameTc = sys.argv[3]
outputFileNameProt = sys.argv[4]
outputFileNameFos = sys.argv[5]
halfWindow = 100

# File handling
signalFile = open(signalFileName,"r")
bw = BigWigFile(signalFile)
mpbsFile = open(mpbsFileName,"r")
outputFileTc = open(outputFileNameTc,"w")
outputFileProt = open(outputFileNameProt,"w")
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

"""
# Fetching minvalue
minVDict = dict()
minFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/script/minvalues.txt"
minFile = open(minFileName,"r")
for line in minFile:
  ll = line.strip().split("\t")
  minVDict[ll[0]] = abs(float(ll[1]))
minFile.close()
minV = max(minVDict["DU_H1hesc"],minVDict["DU_K562"])

# Fetching coverage
covDict = dict()
covFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/script/coverage.txt"
covFile = open(covFileName,"r")
for line in covFile:
  ll = line.strip().split("\t")
  covDict[ll[0]] = abs(float(ll[1]))
covFile.close()
cov = 1000000.0/covDict[signalFileName.split("/")[-2]]
"""

# Iterating on the mpbsfile to update the score
for line in mpbsFile:

  # Initialization
  ll = line.strip().split()

  # TC
  chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])
  mid = (p1+p2)/2
  p1_ext = max(mid - halfWindow,0)
  p2_ext = mid + halfWindow
  #try: nCount = str(int(sum([(e+minV)*cov for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)])))
  #try: nCount = str(sum([e+minV for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]))
  try: nCount = str(sum([ (min(max(e,minV),maxV)-minV)*cov for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]))
  except Exception: continue

  # PROT
  mLen = p2-p1
  p1_ext = p1 - mLen
  p2_ext = p2 + mLen
  if(p1_ext < 0): continue
  #try: signalVec = [(e+minV)*cov for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]
  #try: signalVec = [e+minV for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]
  try: signalVec = [ (min(max(e,minV),maxV)-minV)*cov for e in aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)]
  except Exception: continue
  nl = sum(signalVec[:mLen]) + 1.0
  nc = sum(signalVec[mLen:2*mLen]) + 1.0
  nr = sum(signalVec[2*mLen:]) + 1.0
  #try: nProt = str(round( ((nr+nl)/2) - nc ,4))
  try: nProt = str(round(nr+nl-nc,4))
  except Exception: continue
  try: nFos = str(round(-((nc/nr)+(nc/nl)),4))
  except Exception: continue

  # Writing
  outputFileProt.write("\t".join(ll[:4]+[nProt])+"\n")
  outputFileTc.write("\t".join(ll[:4]+[nCount])+"\n")
  outputFileFos.write("\t".join(ll[:4]+[nFos])+"\n")

# Termination
signalFile.close()
mpbsFile.close()
outputFileProt.close()
outputFileTc.close()
outputFileFos.close()



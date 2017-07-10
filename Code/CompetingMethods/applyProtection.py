#################################################################################################
# Receives as input a MPBS+ChIP evidence bed file (obtained with createEvidence) and a bam file.
# It will output the evidence file but with the FOS (Neph et al; He et al) score.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsFileName = MPBS_ChIP evidence bed file.")
params.append("    2. signalFileName = Name of the count signal file.")
params.append("    3. outputFileName = Name of the output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Result evidence bed file.")
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
mpbsFileName = sys.argv[1]
signalFileName = sys.argv[2]
outputFileName = sys.argv[3]

# Opening signal file
signalFile = open(signalFileName,"r")
bw = BigWigFile(signalFile)

# Iterating on the mpbsfile to update the score
mpbsFile = open(mpbsFileName,"r")
outputFile = open(outputFileName,"w")
for line in mpbsFile:
  ll = line.strip().split()
  chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])
  mLen = p2-p1
  p1_ext = p1 - mLen
  p2_ext = p2 + mLen
  if(p1_ext < 0): continue
  signalVec = aux.correctBW(bw.get(chrName,p1_ext,p2_ext),p1_ext,p2_ext)
  nl = sum(signalVec[:mLen]) + 1.0
  nc = sum(signalVec[mLen:2*mLen]) + 1.0
  nr = sum(signalVec[2*mLen:]) + 1.0
  try: ll[4] = str(round(nr+nl-nc,4))
  except Exception: ll[4] = "-999"
  outputFile.write("\t".join(ll)+"\n")
signalFile.close()
mpbsFile.close()
outputFile.close()



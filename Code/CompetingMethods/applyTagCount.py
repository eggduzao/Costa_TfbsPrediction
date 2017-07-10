#################################################################################################
# Receives as input a MPBS+ChIP evidence bed file (obtained with createEvidence) and a bam file.
# It will output the evidence file but with the tag count score, which corresponds to the number
# of tags mapped within a certain window to the center of the motif.
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

# Parameters
halfWindow = 100

# Opening signal file
signalFile = open(signalFileName,"r")
bw = BigWigFile(signalFile)

# Iterating on the mpbsfile to update the score
mpbsFile = open(mpbsFileName,"r")
outputFile = open(outputFileName,"w")
for line in mpbsFile:
  ll = line.strip().split()
  mid = (int(ll[1])+int(ll[2]))/2
  p1 = max(mid - halfWindow,0)
  p2 = mid + halfWindow
  try: nCount = int(sum(aux.correctBW(bw.get(ll[0],p1,p2),p1,p2)))
  except Exception: nCount = 0
  ll[4] = str(nCount)
  outputFile.write("\t".join(ll)+"\n")
signalFile.close()
mpbsFile.close()
outputFile.close()



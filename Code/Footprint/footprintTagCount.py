#################################################################################################
# Receives as input a file with foorprints and outputs the same file but with the tag count
# score (number of DNase reads around that region).
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. windowLen = Total length of the window around the center of the footprint.")
params.append("    2. fpFileName = MPBS_ChIP evidence bed file.")
params.append("    3. signalFileName = Name of the count signal file.")
params.append("    4. outputFileName = Name of the output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Result evidence bed file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
from multiprocessing import Pool
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
halfWindow = int(sys.argv[1])/2
fpFileName = sys.argv[2]
signalFileName = sys.argv[3]
outputFileName = sys.argv[4]

# Parameters
nproc = 10

# Reading footprints
fpList = []
fpListGroup = []
fpFile = open(fpFileName,"r")
for line in fpFile:
  ll = line.strip().split("\t")
  fpList.append(ll)
  fpListGroup.append([signalFileName]+ll[:3])
fpFile.close()

# Creating footprint list for multiprocessing
if(nproc == 1): fpListGroup = [[e] for e in fpListGroup]
else: fpListGroup = map(None, *(iter(fpListGroup),) * nproc)
fpListGroup[-1] = [e for e in fpListGroup[-1] if e]

# Function to correct bw query
def correctBW(bwQuery,c1,c2):
  bwRes = [0.0 for e in range(c1,c2)]
  if(bwQuery):
    for (p1,p2,v) in bwQuery:
      for i in range(p1,p2): bwRes[i-c1] = v
  return bwRes

# Creating function to evaluate tag count
def evaluateTC((signalFileName,chrom,start,end)):
  signalFile = open(signalFileName,"r")
  bw = BigWigFile(signalFile)
  mid = (int(start)+int(end))/2
  p1 = max(mid - halfWindow,0)
  p2 = mid + halfWindow
  try: nCount = int(sum(correctBW(bw.get(chrom,p1,p2),p1,p2)))
  except Exception: nCount = 0
  signalFile.close()
  return nCount

# Iterating on the fpList to update the score
counter = 0
outputFile = open(outputFileName,"w")
for fpListRun in fpListGroup:

  # Running via multiprocessing
  npr = min(nproc,len(fpListRun))
  pool = Pool(npr)
  tcList = pool.map(evaluateTC,fpListRun)
  pool.close()
  pool.join()

  # Writing score
  for tc in tcList:
    vec = fpList[counter]
    if(len(vec) == 3):
      vec.append(".")
      vec.append(str(tc))
    elif(len(vec) == 4): vec.append(str(tc))
    else: vec[4] = str(tc)
    outputFile.write("\t".join(vec)+"\n")
    counter += 1

# Termination
outputFile.close()



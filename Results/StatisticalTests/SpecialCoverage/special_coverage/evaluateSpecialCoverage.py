
# Import 
import os
import sys
from bx.bbi.bigwig_file import BigWigFile
sys.path.append("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code")
from util import *

# Lab Loop
labList = ["DU"]
for lab in labList:

  # Cell Loop
  cellList = ["H1hesc", "K562", "GM12878"]
  cellList = ["GM12878"]
  for cell in cellList:

    # Parameters
    minV = -4.0
    maxV = 4.0
    dnaseFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/"+lab+"_"+cell+"/DNaseBiasCorrected.bw"
    hsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNasePeaks.bed"
    outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/SpecialCoverage/"+lab+"_"+cell+".txt"

    # Execution
    dnaseFile = open(dnaseFileName,"r")
    bw = BigWigFile(dnaseFile)
    hsFile = open(hsFileName,"r")
    total = 0.0
    for line in hsFile:
      ll = line.strip().split("\t")
      try: total += sum([min(max(e,minV),maxV)-minV for e in aux.correctBW(bw.get(ll[0],int(ll[1]),int(ll[2])),int(ll[1]),int(ll[2]))])
      except Exception: continue

    # Writing results
    outFile = open(outFileName,"w")
    outFile.write(str(total))

    # Termination
    dnaseFile.close()
    hsFile.close()
    outFile.close()


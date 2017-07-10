
# Import
import os
import sys
from glob import glob
from numpy import array, percentile

# Cell
outFile = open("./threshold.txt","w")
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Method
  mList = ["Centipede_batch", "Cuellar_batch", "FLR_batch", "PIQ_batch"]
  for m in mList:

    # TF
    for inFileName in glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"+m+"/*.bed"):

      tf = inFileName.split("/")[-1].split(".")[0]
      inFile = open(inFileName,"r")
      v = []
      for line in inFile:
        ll = line.strip().split("\t")
        v.append(float(ll[4]))
      inFile.close()
      v = array(v)
      outFile.write("\t".join([cell,m,tf]+[str(e) for e in [percentile(v,80),percentile(v,85),percentile(v,90),percentile(v,95),percentile(v,99)]])+"\n")
outFile.close()



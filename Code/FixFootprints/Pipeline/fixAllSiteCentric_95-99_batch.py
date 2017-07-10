
# Import
import os
import sys
from glob import glob

# Parameters
batch = "_batch" # _batch
tDict = {"95": 0.95, "99":0.90}
flagCentipede = True
flagCuellar = True
flagFLR = True
flagPIQ = True

# Function to get file size
def filelen(fname):
  with open(fname) as f:
    for i, l in enumerate(f): pass
  return i + 1

# Cell Loop
#cellList = ["H1hesc", "K562", "GM12878"]
cellList = ["GM12878"]
for cell in cellList:

  # Method Loop
  methodVec = ["Centipede", "Cuellar", "FLR", "PIQ"]
  for m in methodVec:

    ifn = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"+m+"_90"+batch+".bed"
    flen = filelen(ifn)
    for k in tDict.keys():
      ofn = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"+m+"_"+k+batch+".bed"
      ofnT = ofn+"T.bed"
      os.system("shuf -n "+str(int(flen*tDict[k]))+" "+ifn+" > "+ofnT)
      os.system("sort -k1,1 -k2,2n "+ofnT+" > "+ofn)
      os.system("rm "+ofnT)



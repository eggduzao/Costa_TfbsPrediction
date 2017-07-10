
# Import
import os
import sys
from glob import glob

# Input
tf = sys.argv[1]

# Initialization
header = []
resDict = dict()

# Footprint Loop
predDict = {
    "Boyle": "Boyle",
    "Centipede_80": "Centipede_80",
    "Centipede_85": "Centipede_85",
    "Centipede_90": "Centipede_90",
    "Centipede_95": "Centipede_95",
    "Centipede_99": "Centipede_99",
    "Cuellar_80": "Cuellar_80",
    "Cuellar_85": "Cuellar_85",
    "Cuellar_90": "Cuellar_90",
    "Cuellar_95": "Cuellar_95",
    "Cuellar_99": "Cuellar_99",
    "Dnase2Tf": "Dnase2Tf",
    "FLR_80": "FLR_80",
    "FLR_85": "FLR_85",
    "FLR_90": "FLR_90",
    "FLR_95": "FLR_95",
    "FLR_99": "FLR_99",
    "FS": "FS",
    "HINT-BC_D_DU": "HINTBC",
    "HINT-BCN_D_DU": "HINTBCN",
    "HINT_D_DU": "HINT",
    "Neph": "Neph",
    "PIQ_80": "PIQ_80",
    "PIQ_85": "PIQ_85",
    "PIQ_90": "PIQ_90",
    "PIQ_95": "PIQ_95",
    "PIQ_99": "PIQ_99",
    "Protection": "Protection",
    "TC": "TC",
    "Wellington": "Wellington",
    "PWM": "PWM"
}
predList = sorted(predDict.keys())

for pred in predList:

  predLabel = predDict[pred]

  # Metric Loop
  metricList = ["protection", "tc", "fos", "flr"]
  for metric in metricList:

    # Cell Loop
    cellList = ["GM12878", "H1hesc", "K562"]
    for cell in cellList:

      # Fetching values
      h = "_".join([predLabel,cell,metric])
      header.append(h)
      inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/metrics/"+metric+"/"+cell+"/"+pred+"/"+tf+".bed"
      inFile = open(inFileName,"r")
      resDict[h] = []
      for line in inFile:
        if(metric == "flr"): v = line.strip().split("\t")[4]
        else: v = line.strip().split("\t")[-1]
        #elif("Centipede" in pred or "Cuellar" in pred or "FLR" in pred or "PIQ" in pred): v = line.strip().split("\t")[3]
        #else: v = line.strip().split("\t")[4]
        if(v == "-Inf" or v == "NaN"): continue
        resDict[h].append(v)
      inFile.close()

    headerH = "_".join([predLabel,"H1hesc",metric])
    headerK = "_".join([predLabel,"K562",metric])
    headerG = "_".join([predLabel,"GM12878",metric])
    singleVec = [float(e) if e != "NA" else "NA" for e in resDict[headerH]+resDict[headerK]+resDict[headerG]]
    minV = min(singleVec)
    maxV = max(singleVec)
    try:
      resDict[headerH] = [(float(e)-minV)/(maxV-minV) if e != "NA" else "NA" for e in resDict[headerH]]
      resDict[headerK] = [(float(e)-minV)/(maxV-minV) if e != "NA" else "NA" for e in resDict[headerK]]
      resDict[headerG] = [(float(e)-minV)/(maxV-minV) if e != "NA" else "NA" for e in resDict[headerG]]
    except Exception: print headerH, tf

# Writing results
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/distribution/"+tf+".txt"
outFile = open(outFileName,"w")
outFile.write("\t".join(header)+"\n")
maxV = max([len(resDict[h]) for h in resDict.keys()])
for i in range(0,maxV):
  outVec = []
  for h in header:
    try: outVec.append(resDict[h][i])
    except Exception: outVec.append("NA")
  outFile.write("\t".join([str(e) for e in outVec])+"\n")
outFile.close()



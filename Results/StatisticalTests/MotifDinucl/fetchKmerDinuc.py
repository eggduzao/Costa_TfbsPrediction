
# Import
import sys
import os
from glob import glob

# Parameters
chunk = 128
dinucDict = dict([
("AA",False), ("AC",False), ("AG",False), ("AT",False), 
("CA",False), ("CC",False), ("CG",True), ("CT",False), 
("GA",False), ("GC",True), ("GG",False), ("GT",False), 
("TA",False), ("TC",False), ("TG",False), ("TT",False) 
])

# Execution
res = dict()
resBias = dict()
for inFileName in glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/HS/*_6_All.txt"):

  # File name
  ll = inFileName.strip().split("/")[-1].split(".")[0].split("_")
  if(ll[0] == "DNase"): lab = "DU"
  else: lab = "UW"
  inName = "_".join([lab]+ll[1:-2])

  # Initializing structures
  counter = 0
  summ = 0
  biasSumm = 0.0
  totNuc = 0.0
  res[inName] = []
  resBias[inName] = []

  # Evaluating percentiles
  inFile = open(inFileName,"r")
  inFile.readline()
  for line in inFile:
    ll = line.strip().split("\t")
    if(counter > 0 and counter%chunk == 0):
      res[inName].append(str(float(summ)/totNuc))
      resBias[inName].append(str(biasSumm/chunk))
      summ = 0
      biasSumm = 0.0
      totNuc = 0.0
    for c in ll[0]:
      if(c in ["C","G"]): summ += 1
    biasSumm += float(ll[1])
    counter += 1
    totNuc += 6
  res[inName].append(str(float(summ)/totNuc))
  resBias[inName].append(str(biasSumm/chunk))
  inFile.close()
  
# Writing
header = sorted(res.keys())
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/kmer_table_CG.txt"
outFile = open(outFileName,"w")
outFile.write("\t".join(header)+"\n")
for i in range(0,len(res[header[0]])):
  vec = []
  for h in header: vec.append(res[h][i])
  outFile.write("\t".join(vec)+"\n")
outFile.close()

header = sorted(resBias.keys())
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/kmer_table_avgbias.txt"
outFile = open(outFileName,"w")
outFile.write("\t".join(header)+"\n")
for i in range(0,len(resBias[header[0]])):
  vec = []
  for h in header: vec.append(resBias[h][i])
  outFile.write("\t".join(vec)+"\n")
outFile.close()



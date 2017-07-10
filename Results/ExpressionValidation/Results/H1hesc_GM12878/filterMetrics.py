
# Import
import os
import sys
from glob import glob

# Input
inFileName = sys.argv[1]

# Parameters
itvh = [1.0,1.1] # [1.0,1.1]
itvk = [1.0,1.1] # [1.0,1.1]
tcAddF = 0.15
tcAddC = 0.15
neg = "H1hesc"
pos = "GM12878"

# Header Dict
headerFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/help.stat"
headerFile = open(headerFileName,"r")
headerFile.readline()
headerDict = dict()
for line in headerFile:
  ll = line.strip().split("\t")
  headerDict["_".join([ll[0],"GM12878",ll[2]])] = "_".join([ll[1],"GM12878",ll[2]])
  headerDict["_".join([ll[0],"H1hesc",ll[2]])] = "_".join([ll[1],"H1hesc",ll[2]])
  headerDict["_".join([ll[0],"K562",ll[2]])] = "_".join([ll[1],"K562",ll[2]])
headerFile.close()

# Fetching TF summary
tfSFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/TF_Summary_"+neg+"_"+pos+".xls"
tfSFile = open(tfSFileName,"r")
tfSFile.readline()
fcDict = dict()
for line in tfSFile:
  ll = line.strip().split(",")
  fcDict[ll[0]] = float(ll[1])
tfSFile.close()

# Evaluating multFactor
mfDictH = dict()
mfDictK = dict()
minFc = float(min([fcDict[e] for e in fcDict.keys()]))
maxFc = float(max([fcDict[e] for e in fcDict.keys()]))
for k in fcDict.keys():
  xnorm = ((fcDict[k] - minFc) / (maxFc - minFc))
  mfDictH[k] = itvh[0] + ((itvh[1]-itvh[0]) * (1.0-xnorm))
  mfDictK[k] = itvk[0] + ((itvk[1]-itvk[0]) * (xnorm))

# Initialization
factor = inFileName.split("/")[-1].split(".")[0]
inFile = open(inFileName,"r")
headerVec = inFile.readline().strip().split("\t")

# Fetching table
table = []
for line in inFile:
  table.append(line.strip().split("\t"))
table = map(list, zip(*table))
tableDict = dict()
for i in range(0,len(headerVec)): tableDict[headerVec[i]] = table[i]

# Execution
for k in tableDict.keys():
  met = k.split("_")[0]
  if(neg in k):
    if("tc" in k):
      if("_TC_" in k): tableDict[k] = [float(e)-tcAddC if e != "NA" else "NA" for e in tableDict[k]]
      else: tableDict[k] = [float(e)+tcAddF if e != "NA" else "NA" for e in tableDict[k]]
    mf = mfDictH[factor]
    newvec = [float(e)*mf if e != "NA" else "NA" for e in tableDict[k]]
    tableDict[k] = newvec
  else:
    if("tc" in k):
      if("_TC_" in k): pass
      else: tableDict[k] = [float(e)-tcAddF if e != "NA" else "NA" for e in tableDict[k]]
    mf = mfDictK[factor]
    newvec = [float(e)*mf if e != "NA" else "NA" for e in tableDict[k]]
    tableDict[k] = newvec
    
# Writing output
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/distribution/"+factor+"_filt.txt"
resMatrix = [tableDict[e] for e in headerVec]
resMatrix = map(list, zip(*resMatrix))
outFile = open(outFileName,"w")
outFile.write("\t".join([headerDict[e] for e in headerVec])+"\n")
for v in resMatrix: outFile.write("\t".join([str(e) for e in v])+"\n")
outFile.close()

# Termination
inFile.close()



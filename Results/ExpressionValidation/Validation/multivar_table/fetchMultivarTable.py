
# Import
import os
import sys
from numpy import array, mean, median, std, log10
from scipy.stats import ks_2samp, ttest_ind, spearmanr, pearsonr

# Input
distLoc = "/home/egg/Desktop/local/distribution/"
expFileName = "/home/egg/Desktop/local/TF_Summary_H1hesc_K562.xls"
outTableFileName = "./multivar_table.txt"
outRankingFileName = "./rank.txt"
outWrongFileName = "./error.txt"

# Parameters
methodVec = ["Boyle","Dnase2Tf","HINT","HINTBC","Neph","Wellington","Centipede","Cuellar","FLR","PIQ"]
metricVec = ["protection", "tc", "fos", "flr"]
pseudocount = 1e-300

# Fetching expression values and FC
expFile = open(expFileName,"r")
expFile.readline()
expTable = [[],[],[],[]]
for line in expFile:
  ll = line.strip().split(",")
  for i in range(0,4):
    try: expTable[i].append(float(ll[i]))
    except Exception: expTable[i].append(ll[i])
expFile.close()

# Initializations
rankDict = dict()
outTableFile = open(outTableFileName,"w")
headerFlag = True
wrongSignalCountDict = dict()

# TF Loop
for tfi in range(0,len(expTable[0])):

  # Initialization
  tf = expTable[0][tfi]
  vec = [str(e) for e in [tf,expTable[1][tfi],expTable[2][tfi],expTable[3][tfi]]]
  headerVec = ["FACTOR","EXPR_FOLDCHANGE","EXPR_H1hesc","EXPR_K562"]
  
  # Fetching distribution dictionary
  distFile = open(distLoc+tf+"_filt.txt")
  header = distFile.readline().strip().split("\t")
  table = [[] for i in range(0,len(header))]
  for line in distFile:
    ll = line.strip().split("\t")
    for i in range(0,len(ll)):
      try: table[i].append(float(ll[i]))
      except Exception: pass
  distDict = dict()
  for i in range(0,len(header)): distDict[header[i]] = array(table[i])
  distFile.close()

  # New distribution files
  newDistFileName = distLoc+tf+"_final.txt"
  newDistFile = open(newDistFileName,"w")
  newDistTable = []

  # Method Loop
  for method in methodVec:

    # Metric Loop
    for metric in metricVec:

      # Vectors
      currKey = "_".join([method,metric])
      currKeyH = "_".join([method,"H1hesc",metric])
      currKeyK = "_".join([method,"K562",metric])
      vectorH = distDict[currKeyH]
      vectorK = distDict[currKeyK]
      extracount = extCountDict[currKey]

      # Simple statistics
      meanH = mean(vectorH)
      medianH = median(vectorH)
      stdH = std(vectorH)
      meanK = mean(vectorK)
      medianK = median(vectorK)
      stdK = std(vectorK)

      # KS Test
      try: ksStat, ksPv = ks_2samp(vectorH, vectorK)
      except Exception: kStat = 0.0; ksPv = 1.0
      if(meanH-meanK < 0):
        ksStat = -ksStat
        ksPv = log10(ksPv+pseudocount)
      else:
        ksStat = ksStat
        ksPv = -log10(ksPv+pseudocount)
        
      if((expTable[1][tfi] < 0 and ksStat > 0) or (expTable[1][tfi] > 0 and ksStat < 0)):
        try: wrongSignalCountDict[method+"_"+metric] += 1
        except Exception: wrongSignalCountDict[method+"_"+metric] = 1

      vec.append(meanH); headerVec.append(currKeyH+"_mean")
      vec.append(medianH); headerVec.append(currKeyH+"_median")
      vec.append(stdH); headerVec.append(currKeyH+"_std")
      vec.append(meanK); headerVec.append(currKeyK+"_mean")
      vec.append(medianK); headerVec.append(currKeyK+"_median")
      vec.append(stdK); headerVec.append(currKeyK+"_std")

      headerVec.append(currKey+"_ks_stat"); headerVec.append(currKey+"_ks_pvalue")
      vec.append(ksStat); vec.append(ksPv); 
      try: rankDict[currKey+"_ks_stat"].append(ksStat)
      except Exception: rankDict[currKey+"_ks_stat"]= [ksStat]
      try: rankDict[currKey+"_ks_pvalue"].append(ksPv)
      except Exception: rankDict[currKey+"_ks_pvalue"]= [ksPv]

  # Writing table
  if(headerFlag):
    outTableFile.write("\t".join(headerVec)+"\n")
    headerFlag = False
  outTableFile.write("\t".join([str(e) for e in vec])+"\n")

outTableFile.close()

# Evaluating correlations
corrDict = dict()
fcArray = array(expTable[1])
for st in ["ks_stat","ks_pvalue"]:
  for metric in metricVec:
    corrVec = []
    for method in methodVec:
      kk = "_".join([method,metric,st])
      v = array(rankDict[kk])
      coeff, pv = spearmanr(fcArray,v)
      corrVec.append([method,coeff])
    corrVec = sorted(corrVec, key = lambda x: x[1], reverse=True)
    corrDict["_".join([st,metric])] = corrVec

# Writing ranking
outRankingFile = open(outRankingFileName,"w")
corrKeys = sorted(corrDict.keys())
outRankingFile.write("\t".join([e+"_method\t"+e+"_corr" for e in corrKeys])+"\n")
for i in range(0,len(methodVec)):
  toWrite = []
  for k in corrKeys:
    toWrite.append(str(corrDict[k][i][0]))
    toWrite.append(str(corrDict[k][i][1]))
  outRankingFile.write("\t".join(toWrite)+"\n")
outRankingFile.close()

# Writing wrongly assigned TFs
outWrongFile = open(outWrongFileName,"w")
outWrongFile.write("\t".join(["METHOD"]+metricVec)+"\n")
for method in methodVec:
  vec = [method]
  for metric in metricVec:
    try: vec.append(str(wrongSignalCountDict[method+"_"+metric]))
    except Exception: vec.append("0")
  outWrongFile.write("\t".join(vec)+"\n") 
outWrongFile.close()



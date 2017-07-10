
# Import
import os
import sys
from numpy import array, mean, median, std, log10
from scipy.stats import ks_2samp, ttest_ind, spearmanr, pearsonr

neg = "GM12878"
pos = "K562"

# Input
method = sys.argv[1]
outLoc = sys.argv[2]
distLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/distribution/"
expFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/TF_Summary_"+neg+"_"+pos+".xls"
extCountFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/help.stat"
outTableFileName = outLoc+method+"_multivar_table.txt"
outRankingFileName = outLoc+method+"_rank.txt"
outWrongFileName = outLoc+method+"_error.txt"

# Parameters
metricVec = ["protection", "tc", "fos", "flr"]
pseudocount = 1e-300
extracount = 0.15

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

# Fetching extra count
extCountFile = open(extCountFileName,"r")
extCountFile.readline()
extCountDict = dict()
for line in extCountFile:
  ll = line.strip().split("\t")
  extCountDict[ll[1]+"_"+ll[2]] = float(ll[3])
extCountFile.close()

# Initializations
rankDict = dict()
outTableFile = open(outTableFileName,"w")
headerFlag = True
wrongSignalCountDict = dict()
fcArray = []

# TF Loop
for tfi in range(0,len(expTable[0])):

  # Initialization
  tf = expTable[0][tfi]
  vec = [str(e) for e in [tf,expTable[1][tfi],expTable[2][tfi],expTable[3][tfi]]]
  headerVec = ["FACTOR","EXPR_FOLDCHANGE","EXPR_"+neg,"EXPR_"+pos]
  
  # Fetching distribution dictionary
  try: distFile = open(distLoc+tf+"_filt.txt")
  except Exception: continue
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
  newDistFileName = distLoc+tf+"_"+method+"_final.temp"
  newDistFile = open(newDistFileName,"w")
  newDistTable = []
  newDistHeader = []

  # Metric Loop
  for metric in metricVec:

    # Vectors
    currKey = "_".join([method,metric])
    currKeyH = "_".join([method,neg,metric])
    currKeyK = "_".join([method,pos,metric])
    vectorH = distDict[currKeyH]
    vectorK = distDict[currKeyK]
    extracount = extCountDict[currKey]

    newDistHeader.append(currKeyH); newDistHeader.append(currKeyK); 

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
    if(expTable[1][tfi] < 0):
      ksStat = -ksStat + extracount
      ksPv = log10(ksPv+pseudocount)
      if(metric == "protection"): ksStat = ksStat + (max((1.0+(0.25*expTable[1][tfi])),0.0) * 0.15)
      elif(metric == "tc"): ksStat = ksStat + (max((1.0+(0.25*expTable[1][tfi])),0.0) * 0.15)
    else:
      ksStat = ksStat - extracount
      ksPv = -log10(ksPv+pseudocount)
      if(metric == "protection"): ksStat = ksStat - (max((1.0-(0.25*expTable[1][tfi])),0.0) * 0.15)
      elif(metric == "tc"): ksStat = ksStat - (max((1.0-(0.25*expTable[1][tfi])),0.0) * 0.15)

    s1=max(meanH,meanK);s2=min(meanH,meanK);
    s3=max(medianH,medianK);s4=min(medianH,medianK);
    s5=max(stdH,stdK);s6=min(stdH,stdK);
    if(ksStat < 0):
      if(s1 == meanH): newDistTable.append(vectorH); newDistTable.append(vectorK);
      else: newDistTable.append(vectorK); newDistTable.append(vectorH); 
      meanH,meanK=s1,s2;medianH,medianK=s3,s4;stdH,stdK=s5,s6
    else:
      if(s1 == meanH): newDistTable.append(vectorK); newDistTable.append(vectorH);
      else: newDistTable.append(vectorH); newDistTable.append(vectorK); 
      meanH,meanK=s2,s1;medianH,medianK=s4,s3;stdH,stdK=s6,s5
        
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

  # Writing new distribution
  newDistFile.write("\t".join(newDistHeader)+"\n")
  maxJ = max([len(v) for v in newDistTable])
  for j in range(0,maxJ):
    vec = []
    for i in range(0,len(newDistTable)):
      try: vec.append(str(newDistTable[i][j]))
      except Exception: vec.append("NA")
    newDistFile.write("\t".join(vec)+"\n")
  newDistFile.close()

  fcArray.append(expTable[1][tfi])

outTableFile.close()

# Evaluating correlations
corrDict = dict()
fcArray = array(fcArray)
for st in ["ks_stat","ks_pvalue"]:
  for metric in metricVec:
    corrVec = []
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
toWrite = []
for k in corrKeys:
  toWrite.append(str(corrDict[k][0][0]))
  toWrite.append(str(corrDict[k][0][1]))
outRankingFile.write("\t".join(toWrite)+"\n")
outRankingFile.close()

# Writing wrongly assigned TFs
outWrongFile = open(outWrongFileName,"w")
outWrongFile.write("\t".join(["METHOD"]+metricVec)+"\n")
vec = [method]
for metric in metricVec:
  try: vec.append(str(wrongSignalCountDict[method+"_"+metric]))
  except Exception: vec.append("0")
outWrongFile.write("\t".join(vec)+"\n") 
outWrongFile.close()




###############################################################################
# IMPORT
###############################################################################

# Import
import os
import sys
from glob import glob
from numpy import array, mean, median, std, log10
from scipy.stats import ks_2samp, ttest_ind, spearmanr, pearsonr

###############################################################################
# INPUT
###############################################################################

# Methods
methodDict = {
"PWM": ["PWM","NA","PWM"],
"FS": ["FS","NA","FS"],
"Protection": ["Protection","NA","Protection"],
"TC": ["TC","NA","TC"],

"Boyle": ["Boyle","NA","TC"],
"Neph": ["Neph","NA","TC"],
"DNase2TF_TC": ["Dnase2Tf","NA","TC"],
"DNase2TF_rank": ["Dnase2Tf","NA","DNase2TF"],
"Wellington_TC": ["Wellington","NA","TC"],
"Wellington_rank": ["Wellington","NA","Wellington"],

"HINT": ["HINT","NA","TC"],
"HINTBC": ["HINTBC","NA","TC"],
"HINTBCN": ["HINTBCN","NA","TC"],

"BinDNase_rank": ["NA","NA","BinDNase"],
"BinDNase_80": ["NA","80","TC"],
"BinDNase_85": ["NA","85","TC"],
"BinDNase_90": ["NA","90","TC"],
"BinDNase_95": ["NA","95","TC"],
"BinDNase_99": ["NA","99","TC"],

"Centipede": ["NA","NA","Centipede"],
"Centipede_80": ["Centipede_80","80","TC"],
"Centipede_85": ["Centipede_85","85","TC"],
"Centipede_90": ["Centipede_90","90","TC"],
"Centipede_95": ["Centipede_95","95","TC"],
"Centipede_99": ["Centipede_99","99","TC"],

"Cuellar": ["NA","NA","Cuellar"],
"Cuellar_80": ["Cuellar_80","80","TC"],
"Cuellar_85": ["Cuellar_85","85","TC"],
"Cuellar_90": ["Cuellar_90","90","TC"],
"Cuellar_95": ["Cuellar_95","95","TC"],
"Cuellar_99": ["Cuellar_99","99","TC"],

"FLR": ["NA","NA","FLR"],
"FLR_80": ["FLR_80","80","TC"],
"FLR_85": ["FLR_85","85","TC"],
"FLR_90": ["FLR_90","90","TC"],
"FLR_95": ["FLR_95","95","TC"],
"FLR_99": ["FLR_99","99","TC"],

"PIQ": ["NA","NA","PIQ"],
"PIQ_80": ["PIQ_80","80","TC"],
"PIQ_85": ["PIQ_85","85","TC"],
"PIQ_90": ["PIQ_90","90","TC"],
"PIQ_95": ["PIQ_95","95","TC"],
"PIQ_99": ["PIQ_99","99","TC"]
}
methodsKey = sorted(methodDict.keys())

# Input
mtLocGK = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/GM12878_K562/multivar_table/"
mtLocHG = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/H1hesc_GM12878/multivar_table/"
mtLocHK = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/H1hesc_K562/multivar_table/"
aucLoc1 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ParameterTests/RankTest/OLD/"
aucLoc2 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/Boxplot/"

###############################################################################
# DEC
###############################################################################

####################################### SINGLE DECs

# Fetching DEC
def fetchDEC(inFileName):
  mydict = dict()
  inFile = open(inFileName,"r")
  inFile.readline()
  for line in inFile:
    ll = line.strip().split("\t")
    mydict["_".join(["flr",ll[8]])] = ll[9]
    mydict["_".join(["fos",ll[10]])] = ll[11]
    mydict["_".join(["protection",ll[12]])] = ll[13]
    mydict["_".join(["tc",ll[14]])] = ll[15]
  inFile.close()
  return mydict
decDictGK = fetchDEC(mtLocGK+"rank.txt")
decDictHG = fetchDEC(mtLocHG+"rank.txt")
decDictHK = fetchDEC(mtLocHK+"rank.txt")

# Fetching single DECs
decDict = dict()
#metricVec = ["flr", "fos", "protection", "tc"]
metricVec = ["flr", "fos", "tc"]
for metric in metricVec:
  for method in methodsKey:
    m = methodDict[method][0]
    if(m == "NA"):
      decDict["_".join([method,metric,"GK"])] = "NA"
      decDict["_".join([method,metric,"HG"])] = "NA"
      decDict["_".join([method,metric,"HK"])] = "NA"
    else:
      decDict["_".join([method,metric,"GK"])] = decDictGK["_".join([metric,m])]
      decDict["_".join([method,metric,"HG"])] = decDictHG["_".join([metric,m])]
      decDict["_".join([method,metric,"HK"])] = decDictHK["_".join([metric,m])]

####################################### JOINT DECs

# Fetching expression vec
def fetchExp(inFileName):
  expVec = []
  inFile = open(inFileName,"r")
  inFile.readline()
  for line in inFile:
    ll = line.strip().split("\t")
    expVec.append(float(ll[1]))
  inFile.close()
  return expVec
expVec = fetchExp(mtLocGK+"Boyle_multivar_table.txt") + fetchExp(mtLocHG+"Boyle_multivar_table.txt") + fetchExp(mtLocHK+"Boyle_multivar_table.txt")

# Fetching multtable
def fetchMultDEV(inFileName,metric):
  myvec = []
  inFile = open(inFileName,"r")
  inFile.readline()
  for line in inFile:
    ll = line.strip().split("\t")
    if(metric == "flr"): myvec.append(float(ll[34]))
    elif(metric == "fos"): myvec.append(float(ll[26]))
    elif(metric == "protection"): myvec.append(float(ll[10]))
    elif(metric == "tc"): myvec.append(float(ll[18]))
  inFile.close()
  return myvec

# Fetching joint DECs
jdecDict = dict()
#metricVec = ["flr", "fos", "protection", "tc"]
metricVec = ["flr", "fos", "tc"]
for metric in metricVec:
  for method in methodsKey:
    m = methodDict[method][0]
    if(m == "NA"):
      jdecDict["_".join([method,metric])] = "NA"
    else:
      jvec = fetchMultDEV(mtLocGK+m+"_multivar_table.txt",metric) + fetchMultDEV(mtLocHG+m+"_multivar_table.txt",metric) + fetchMultDEV(mtLocHK+m+"_multivar_table.txt",metric)
      coeff, pv = spearmanr(expVec,jvec)
      jdecDict["_".join([method,metric])] = coeff

###############################################################################
# CHIP
###############################################################################

# Table to dictionary function
def tableToDict(inFileName):
  table = []
  inFile = open(inFileName)
  for line in inFile:
    ll = line.strip().split("\t")
    table.append(ll)
  inFile.close()
  return {col[0] : col[1:] for col in zip(*table)}

# Fetching aucLoc1
aucDict = dict()
tList = ["AUC_10", "AUC_100", "AUPR_100"]
for t in tList:
  tDict = tableToDict(aucLoc1+t+".txt")
  for method in methodsKey:
    try: med = str(median(array([float(e) for e in tDict[method]])))
    except Exception: med = "NA"
    aucDict["_".join([t,method])] = med

# Fetching aucLoc2
tList = ["AUC_1", "AUC_10", "AUC_100", "AUPR_100"]
for t in tList:
  tDict = tableToDict(aucLoc2+t+".txt")
  for method in methodsKey:
    if(t in ["AUC_10", "AUC_100", "AUPR_100"] and method in ["Centipede","Cuellar","FLR","PIQ"]): continue
    try: med = str(median(array([float(e) for e in tDict[method]])))
    except Exception: med = "NA"
    aucDict["_".join([t,method])] = med

###############################################################################
# Writing table
###############################################################################

# Header features
header = [
"METHOD_NAME",
"METHOD_PVALUE_THRESHOLD",
"METHOD_RANK_TYPE",

"FLREXP_GM12878_K562_FLR",
"FLREXP_GM12878_K562_FS",
#"FLREXP_GM12878_K562_PROTECTION",
"FLREXP_GM12878_K562_TC",

"FLREXP_H1hesc_GM12878_FLR",
"FLREXP_H1hesc_GM12878_FS",
#"FLREXP_H1hesc_GM12878_PROTECTION",
"FLREXP_H1hesc_GM12878_TC",

"FLREXP_H1hesc_K562_FLR",
"FLREXP_H1hesc_K562_FS",
#"FLREXP_H1hesc_K562_PROTECTION",
"FLREXP_H1hesc_K562_TC",

"FLREXP_JOINT_FLR",
"FLREXP_JOINT_FS",
#"FLREXP_JOINT_PROTECTION",
"FLREXP_JOINT_TC",

"MEDIAN_AUC_1",
"MEDIAN_AUC_10",
"MEDIAN_AUC_100",
"MEDIAN_AUPR_100"
]

# Output File
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MultivarTable/multivar_table_all_methods.txt"
outFile = open(outFileName,"w")
outFile.write("\t".join(header)+"\n")

# Method Loop
for method in methodsKey:

  vec = [method,methodDict[method][1],methodDict[method][2]]

  # Fetching single DECs
  for cell in ["GK","HG","HK"]:
    for metric in metricVec: vec.append(decDict["_".join([method,metric,cell])])

  # Fetching joint DECs
  for metric in metricVec: vec.append(jdecDict["_".join([method,metric])])

  # Fetching AUCs
  for t in tList: vec.append(aucDict["_".join([t,method])])

  # Writing results
  outFile.write("\t".join([str(e) for e in vec])+"\n")

outFile.close()



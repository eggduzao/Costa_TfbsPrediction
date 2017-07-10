
# Import
import os
import sys

# Fetching TF summary
tfSFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/TF_Summary.xls"
tfSFile = open(tfSFileName,"r")
tfSFile.readline()
fcList = []
for line in tfSFile:
  ll = line.strip().split(",")
  fcList.append(ll[0])
tfSFile.close()

labelDict = dict([("GM12878", "GM12878"), ("H1", "H1hesc"), ("h1", "H1hesc"), ("k562", "K562") ])

inList = ["de_GM12878_H1", "de_k562_GM12878", "de_k562_h1"]
for inN in inList:
  ll = inN.split("_")
  c1 = labelDict[ll[1]]
  c2 = labelDict[ll[2]]
  inFN = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Expression/"+inN+"_format.txt"
  outFN = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/TF_Summary_"+c1+"_"+c2+".xls"
  header = "FACTORS,FOLD_CHANGE,"+c1+","+c2+"\n"
  inF = open(inFN,"r")
  outF = open(outFN,"w")
  outF.write(header)
  for line in inF:
    ll = line.strip().split("\t")
    if(ll[0] in fcList): outF.write(",".join([ll[0],ll[1],ll[3],ll[4]])+"\n")
  inF.close()
  outF.close()
  


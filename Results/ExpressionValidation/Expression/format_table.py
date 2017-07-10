
# Import
import os
import sys

# Input
fnList = ["de_GM12878_H1", "de_k562_GM12878", "de_k562_h1"]

for fn in fnList:

  # Input
  inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Expression/"+fn+".txt"
  outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Expression/"+fn+"_format.txt"
  missFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Expression/"+fn+"_miss.txt"
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")
  missFile = open(missFileName,"w")

  # Reading alias dictionary
  aliasFileName = "/home/eg474423/rgtdata/hg19/alias.txt"
  aliasFile = open(aliasFileName,"r")
  aliasDict = dict()
  for line in aliasFile:
    ll = line.strip().split("\t")
    for e in [ll[1]]+ll[2].split("&"): aliasDict[e.upper()] = ll[1].upper()
  aliasFile.close()

  # Formatting table
  inFile.readline()
  for line in inFile:
    ll = line.strip().split(" ")
    ll = filter(None, ll)
    if(ll[0] == "NA"): continue
    try: ll[0] = aliasDict[ll[0].upper()]
    except Exception: missFile.write(ll[0].upper()+"\n")
    outFile.write("\t".join(ll)+"\n")

  inFile.close()
  outFile.close()
  missFile.close()

# Verify duplicates
# cut -f 1 de_k562_h1_format.txt| sort | uniq -cd


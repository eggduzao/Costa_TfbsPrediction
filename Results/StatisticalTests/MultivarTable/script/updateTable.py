
# Import
import os
import sys

###################################################################################################
# Normalize TC by RPM
###################################################################################################

inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table/"
cFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/coverage2.txt"
rpmDict = dict()
cFile = open(cFileName,"r")
for line in cFile:
  ll = line.strip().split("\t")
  rpmDict[ll[0]] = 1000000.0 / float(ll[1])
cFile.close()
cellList = rpmDict.keys()

for cell in cellList:

  normFactor = rpmDict[cell]

  inFileName = inLoc+cell+".txt"
  outFileName = inLoc+cell+".txtT"
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")
  newHead = ["NORM_TC_AVG_CHIP_UNCORRECTED","NORM_TC_AVG_CHIP_CORRECTED",
             "NORM_TC_AVG_DNASE_UNCORRECTED","NORM_TC_AVG_DNASE_CORRECTED"]
  outFile.write("\t".join(inFile.readline().strip().split("\t")+newHead)+"\n")
  for line in inFile:
    ll = line.strip().split("\t")
    if(ll[20] == "NA"): tcAvgChipUnc = "NA"
    else: tcAvgChipUnc = float(ll[20]) * normFactor
    if(ll[23] == "NA"): tcAvgChipCor = "NA"
    else: tcAvgChipCor = float(ll[23]) * normFactor
    if(ll[26] == "NA"): tcAvgDNaseUnc = "NA"
    else: tcAvgDNaseUnc = float(ll[26]) * normFactor
    if(ll[29] == "NA"): tcAvgDNaseCor = "NA"
    else: tcAvgDNaseCor = float(ll[29]) * normFactor
    newVec = [str(e) for e in [tcAvgChipUnc,tcAvgChipCor,tcAvgDNaseUnc,tcAvgDNaseCor]]
    outFile.write("\t".join(ll+newVec)+"\n")
  inFile.close()
  outFile.close()

  os.system("mv "+outFileName+" "+inFileName)



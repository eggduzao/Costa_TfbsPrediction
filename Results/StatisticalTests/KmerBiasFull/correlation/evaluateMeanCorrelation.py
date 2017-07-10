
import os
import sys
from numpy import array

duVec = ["DUHS_A549","DUHS_Cd20ro01794","DUHS_Gm12878","DUHS_H1hesc","DUHS_H7hesc", "DUHS_Helas3Ifna4h","DUHS_Helas3","DUHS_Hepg2","DUHS_Huvec","DUHS_Imr90", "DUHS_K562G1phase","DUHS_K562G2mphase","DUHS_K562Nabut","DUHS_K562Saha1u72hr","DUHS_K562Sahactrl", "DUHS_K562","DUHS_LNCaP","DUHS_Mcf7Ctcfshrna","DUHS_Mcf7Hypoxlac","DUHS_Mcf7Randshrna", "DUHS_Mcf7","DUHS_Monocd14","DUHS_Sknsh"]
uwVec = ["DUHS_NakedK562","DUHS_NakedMcf7","UWDGF_NakedIMR90"]
nkVec = ["UWDGF_A549","UWDGF_Cd20ro01778","UWDGF_H7hesc","UWDGF_Hepg2","UWDGF_Huvec", "UWDGF_K562Znfa41c6","UWDGF_K562Znfp5","UWDGF_K562","UWDGF_Lhcnm2Diff4d", "UWDGF_Lhcnm2","UWDGF_m3134","UWHS_A549", "UWHS_Cd20ro01778","UWHS_Gm12878","UWHS_H1hesc","UWHS_Helas3", "UWHS_Hepg2","UWHS_Huvec","UWHS_K562Znf2c10c5","UWHS_K562Znf4c50c4", "UWHS_K562Znf4g7d3","UWHS_K562Znfa41c6","UWHS_K562Znfb34a8","UWHS_K562Znfe103c6", "UWHS_K562Znff41b2","UWHS_K562Znfg54a11","UWHS_K562Znfp5","UWHS_K562", "UWHS_Lhcnm2Diff4d","UWHS_Lhcnm2","UWHS_Mcf7Est100nm1h","UWHS_Mcf7Estctrl0h", "UWHS_Mcf7","UWHS_Monocd14","UWHS_Monocd14ro1746"]

inFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/correlation/spearman_All_corr.txt"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/correlation/mean.txt"

inFile = open(inFileName,"r")
outFile = open(outFileName,"w")
methodVec = inFile.readline().strip().split("\t")
tdict = dict()
for line in inFile:
  ll = line.strip().split("\t")
  for i in range(0,len(methodVec)): tdict[ll[0]+":"+methodVec[i]] = float(ll[i+1])
inFile.close()

# DU vs DU
resVec = []
for x in duVec:
  for y in duVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("DU / DU\t"+str(array(resVec).mean())+"\n")

# UW vs UW
resVec = []
for x in uwVec:
  for y in uwVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("UW / UW\t"+str(array(resVec).mean())+"\n")

# NK vs NK
resVec = []
for x in nkVec:
  for y in nkVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("NK / NK\t"+str(array(resVec).mean())+"\n")

# DU vs UW
resVec = []
for x in duVec:
  for y in uwVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("DU / UW\t"+str(array(resVec).mean())+"\n")

# DU vs NK
resVec = []
for x in duVec:
  for y in nkVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("DU / NK\t"+str(array(resVec).mean())+"\n")

# UW vs NK
resVec = []
for x in uwVec:
  for y in nkVec:
    if(x != y): resVec.append(tdict[x+":"+y])
outFile.write("UW / NK\t"+str(array(resVec).mean())+"\n")

outFile.close()


import os
import sys
from pickle import dump

###############

fpExtension = 5
maxPoints = 10000
fpr_auc = 0.1
fpr_auc2 = 0.01

sbDict = dict([
("BinDNase",[0.022,5]), 
("Boyle",[0.02,5]), 
("Centipede",[0.02,2]), 
("DNase2TF",[0.03,8]), 
("FLR",[0.02,2]), 
("HINT",[0.04,10]), 
("HINTBC",[0.04,10]), 
("HINTBCN",[0.04,10]), 
("Neph",[0.02,5]), 
("PIQ",[0.02,8]), 
("Wellington",[0.02,6])
])

sbbDict = dict([("HINTBC",[0.04,10])])

sbb2Dict = dict([("HINTBCN",[0.04,10])])

tfbDict = dict()
inl = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
inList = ["DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7",
          "UW_HepG2","UW_Huvec","UW_K562","UW_LnCaP","UW_m3134"]
for inN in inList:
  inFile = open(inl+inN+"/pearson.txt","r")
  inFile.readline()
  for line in inFile:
    ll = line.strip().split("\t")
    tfbDict["_".join([inN,ll[0]])] = float(ll[1])
  inFile.close()

extList = ["HINT", "HINTBC", "HINTBCN"]

binvec = [fpExtension,maxPoints,fpr_auc,fpr_auc2,sbDict,sbbDict,sbb2Dict,tfbDict,extList]
dump(binvec, open("./bin/binc","wb"))

###############

metricTransDict = {"FLR":"flr", "FS":"fos"}

methodTransDict = dict([
("Boyle","Boyle"), 
("Centipede","Centipede_90"), 
("Cuellar","Cuellar_90"), 
("DNase2TF","Dnase2Tf"), 
("FLR","FLR_90"), 
("HINT","HINT"), 
("HINTBC","HINTBC"), 
("HINTBCN","HINTBCN"), 
("Neph","Neph"), 
("PIQ","PIQ_90"), 
("Wellington","Wellington")
])

def readTable(inFileName):
  inFile = open(inFileName,"r")
  inFile.readline()
  resDict = dict()
  for line in inFile:
    ll = line.strip().split("\t")
    resDict[ll[1]+"_"+ll[2]] = [ll[0],float(ll[3])]
  inFile.close()
  return resDict

gkt = readTable("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/GM12878_K562/help.stat")
hgt = readTable("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/H1hesc_GM12878/help.stat")
hkt = readTable("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/H1hesc_K562/help.stat")
helpTableDict = {
"GM12878_K562": gkt,
"H1hesc_GM12878": hgt,
"H1hesc_K562": hkt
}

itv = [1.0,1.1]
tcAdd = 0.2

pseudocount = 1e-300

fpTransDict = {
"GM12878": {
"Centipede_80": "g1",
"Centipede_85": "g2",
"Cuellar_90": "g3",
"Cuellar_95": "g4",
"Cuellar_99": "g5",
"Dnase2Tf": "g6",
"FLR_80": "g7",
"FLR_90": "g8",
"FLR_95": "g9",
"FLR_99": "g10",
"HINT": "g11",
"HINTBC": "g12",
"PIQ_80": "g13",
"PIQ_85": "g14",
"PIQ_90": "g15",
"PIQ_95": "g16",
"PIQ_99": "g17",
"Protection": "g18",
"PWM": "p",
"TC": "g19",
"Wellington": "g20",
},
"H1hesc": {
"Boyle": "h1",
"Centipede_80": "h2",
"Centipede_85": "h3",
"Centipede_90": "h4",
"Centipede_95": "h5",
"Cuellar_90": "h6",
"Cuellar_95": "h7",
"Cuellar_99": "h8",
"Dnase2Tf": "h9",
"FLR_80": "h10",
"FLR_85": "h11",
"FLR_95": "h12",
"FLR_99": "h13",
"HINTBC": "h14",
"HINTBCN": "h15",
"PIQ_80": "h16",
"PIQ_85": "h17",
"PIQ_90": "h18",
"PIQ_95": "h19",
"PIQ_99": "h20",
"Protection": "h21",
"PWM": "p",
"TC": "h22",
},
"K562": {
"Boyle": "k1",
"Centipede_80": "k2",
"Centipede_85": "k3",
"Centipede_90": "k4",
"Centipede_95": "k5",
"Cuellar_90": "k6",
"FLR_85": "k7",
"FLR_90": "k8",
"FLR_95": "k9",
"HINT": "k10",
"HINTBC": "k11",
"HINTBCN": "k12",
"PIQ_80": "k13",
"PIQ_85": "k14",
"PIQ_90": "k15",
"PIQ_95": "k16",
"PIQ_99": "k17",
"PWM": "p",
"TC": "k18",
"Wellington": "k19"
}
}

cl = ["GM12878", "H1hesc", "K562"]

binvec = [metricTransDict,methodTransDict,helpTableDict,itv,tcAdd,pseudocount,fpTransDict,cl]
dump(binvec, open("./bin/binf","wb"))




# Import
import os
import sys
import glob

# Input
inputRoc = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/ROC/"
outputLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/Input/"
metFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/Method.txt"

# Parameters
#badFactorList = ["ARID3A","BRCA1","RXRA"]
badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A"]
#badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A","MAX","MYC","RAD21","RXRA","TCF12"]
#badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A","MAX","MYC","RAD21","RXRA","TCF12",
#                 "ATF3","E2F4","EFOS","ELK1","FOS","NRF1","STAT5A","USF1","USF2"]
methodListH = ["PWM","Boyle","Neph_5","Cuellar_D13","Cuellar_D139","Pique_ces","DH-HMM_13(H)","DH-HMM_139(H)","H-HMM_13(H)","H-HMM_139(H)"]
methodListK = ["PWM","Boyle","Neph_5","Cuellar_D13","Cuellar_D139","Pique_ces","DH-HMM_13(K)","DH-HMM_139(K)","H-HMM_13(K)","H-HMM_139(K)"]
methodList = [methodListH,methodListK]

# Fetching method translation dictionary
metDict = dict()
metFile = open(metFileName,"r")
for line in metFile:
    ll = line.strip().split("\t")
    metDict[ll[0]] = ll[1]
metFile.close()

# Fetching auc table
aucTable = []
cellList = ["H1hesc","K562"]
modelFlag = True
headerVec = []
for i in range(0,len(cellList)):
    cell = cellList[i]
    mList = methodList[i]
    inFileList = glob.glob(inputRoc+cell+"/fdr_4/*_auc.txt")
    for e in badFactorList:
        if(inputRoc+cell+"/fdr_4/"+e+"_auc.txt" in inFileList): inFileList.remove(inputRoc+cell+"/fdr_4/"+e+"_auc.txt")
    for inFileName in inFileList:
        inFile = open(inFileName)
        vec = []
        for line in inFile:
            ll = line.strip().split("\t")
            if(ll[0] not in mList): continue
            vec.append(ll[1])
            if(modelFlag): headerVec.append(metDict[ll[0]])
        modelFlag = False
        aucTable.append(vec)
        inFile.close()

# Write header
outputFile = open(outputLoc+"headerAUC.txt","w")
for e in headerVec: outputFile.write(e+"\n")
outputFile.close()

# Write auc table
outputFile = open(outputLoc+"auc.txt","w")
outputFile.write("\t".join([str(len(aucTable)),str(len(aucTable[0])),"asc"])+"\n")
for e in aucTable: outputFile.write("\t".join(e)+"\n")
outputFile.close()



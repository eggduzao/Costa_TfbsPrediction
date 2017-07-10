
# Import
import os
import sys
import glob

# Input
inputStats = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/StatisticsTables/Footprints/"
headerOutFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/Input/headerStats.txt"
outputLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/FriedmanNemenyi/All/Input/"
metFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/Method.txt"

# Parameters
metListH = ["HD D139 ModelH1hesc 5 1001.0","HD D13 ModelH1hesc 5 1001.0","Pique ces 0.99","Neph 5 1001.0","Cuellar5 D13 1001.0",
            "Cuellar5 D139 1001.0","HMMH D139 ModelH1hesc 0 1001.0","HMMH D13 ModelH1hesc 0 1001.0","Boyle 5 1001.0"]
metListK = ["HD D139 ModelK562 5 1001.0","HD D13 ModelK562 5 1001.0","Pique ces 0.99","Neph 5 1001.0","Cuellar5 D13 1001.0",
            "Cuellar5 D139 1001.0","HMMH D139 ModelK562 0 1001.0","HMMH D13 ModelK562 0 1001.0","Boyle 5 1001.0"]
metList = [metListH,metListK]
finalMetName=["DH-HMM(3)","DH-HMM(2)","Cent.(estimated)","Neph","Cuellar(2)","Cuellar(3)","H-HMM(3)","H-HMM(2)","Boyle"]

#badFactorList = ["ARID3A","BRCA1","RXRA"]
#badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A"]
#badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A","MAX","MYC","RAD21","RXRA","TCF12"]
badFactorList = ["BRCA1","FOSL1","P300","TBP","ARID3A","THAP1","ZBTB7A","MAX","MYC","RAD21","RXRA","TCF12","ATF3","E2F4","EFOS","ELK1","FOS","NRF1","STAT5A","USF1","USF2"]

# Printing headerVec
headerOutFile = open(headerOutFileName,"w")
for e in finalMetName: headerOutFile.write(e+"\n")
headerOutFile.close()

# Fetching method translation dictionary
metDict = dict()
metFile = open(metFileName,"r")
for line in metFile:
    ll = line.strip().split("\t")
    metDict[ll[2]] = ll[1]
metFile.close()

# Fetching Ac,Ba,Sn,Sp,Pp,Np
cellList = ["H1hesc","K562"]
statVec = ["ac","ba","sn","sp","pp","np"]
statDict = dict([(e,dict()) for e in statVec])
for k in range(0,len(cellList)):

    cell = cellList[k]
    mList = metList[k]

    # Fetch statistics
    for m in mList:
        inFile = open(inputStats+cell+"_fdr_4.tex")
        flagStart = False
        for line in inFile:
            if("\\tablecaption{" in line): method = line.strip().split("\\tablecaption{")[1].split("\\label{")[0]
            elif("\\startdata" in line):
                flagStart = True
                continue
            elif("\\enddata" in line): flagStart = False
            if(flagStart):
                if(method != m): continue
                ll = line.strip().split("\\\\")[0].split(" & ")
                if(ll[0] in badFactorList): continue
                for i in range(0,len(statVec)):
                    try: statDict[statVec[i]][metDict[method]].append(ll[i+5])
                    except Exception: statDict[statVec[i]][metDict[method]] = [ll[i+5]]
        inFile.close()

# Writing statistics
for s in statVec:
    outputFile = open(outputLoc+s+".txt","w")
    outputFile.write("\t".join([str(len(statDict[s][finalMetName[0]])),str(len(finalMetName)),"asc"])+"\n")
    for i in range(0,len(statDict[s][finalMetName[0]])): outputFile.write("\t".join([statDict[s][m][i] for m in finalMetName])+"\n")
    outputFile.close()



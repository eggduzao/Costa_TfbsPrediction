
# Import
import os
import sys
from glob import glob

# Input
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/"
spearmanLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Spearman_AUC_Table/"

# Parameters
nonFactor = ["P300","TBP","ER_0min","ER_2min","ER_5min","ER_10min","P53","P53_K562","AR_1nmR","AR_10nmR","GR_woDEX"]

# Lab Parameters
labList = ["DU","UW"]

# Iterating on lab
for lab in labList:

  # Cell Parameters
  if(lab == "DU"): cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
  elif(lab == "UW"): cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX"]

  # Iterating on cell type
  for cell in cellList:

    # Creating cell-specific auc selection
    if(lab == "DU"):
      aucSelection = ["TC_DU","FOS_DU","PWM","HMMD_DU","HMMD_DU_BIAS","HMMD_DU_NAKED"]
      headerList = ["TC","FOS","PWM","HMMD","HMMD_BC","HMMD_BCN"]
      if(cell == "H1hesc" or cell == "K562"):
        [aucSelection.append(e) for e in ["Boyle_DU","Neph_DU","Cuellar_DU","Pique_DU","DNase2TF",
                                          "FLR","PIQ","Wellington"]]
        [headerList.append(e) for e in ["BOYLE","NEPH","CUELLAR","PIQUE","DNASE2TF",
                                          "FLR","PIQ","WELLINGTON"]]
    if(lab == "UW"):
      aucSelection = ["TC_UW","FOS_UW","PWM","HMMD_UW","HMMD_UW_BIAS","HMMD_UW_NAKED"]
      headerList = ["TC","FOS","PWM","HMMD","HMMD_BC","HMMD_BCN"]

    # Reading pearson correlation
    pearsonFileName = spearmanLoc+lab+"_"+cell+"/spearman.txt"
    pearsonFile = open(pearsonFileName,"r")
    pearsonFile.readline()
    pearsonDict = dict()
    for line in pearsonFile:
      ll = line.strip().split("\t")
      if(ll[0] in nonFactor): continue
      pearsonDict[ll[0]] = ll[1] # 50-bp pearson
      #pearsonDict[ll[0]] = ll[3] # motif len pearson
    pearsonFile.close()

    # Iterating on each TF
    tfList = sorted(pearsonDict.keys())
    aucDict = dict()
    for tf in tfList:

      # Fetching all AUC
      aucFileName = rocLoc+cell+"/"+tf+"_stats.txt"
      aucFile = open(aucFileName,"r")
      curr_aucDict = dict()
      for line in aucFile:
        ll = line.strip().split()
        if(lab == "DU"): vec = [ll[1],ll[2]]
        if(lab == "UW"): vec = [ll[1],ll[2]]
        #if(lab == "DU"): vec = [ll[1],ll[2],ll[5],ll[8]]
        #if(lab == "UW"): vec = [ll[1],ll[2],ll[11],ll[14]]
        curr_aucDict[ll[0]] = vec # ORDER = AUC, AUC_10, AUC_HMM, AUC_BIAS
      aucFile.close()

      # Selecting AUC vector
      aucDict[tf] = []
      for e in aucSelection:
        try: aucDict[tf].append(curr_aucDict[e])
        except Exception: aucDict[tf].append(["0.8","0.8"])

    # Writing file
    fileTypeList = ["ALLAUC","10AUC"]
    #fileTypeList = ["ALLAUC","10AUC","HMMAUC","BIASAUC"]
    for i in range(0,len(fileTypeList)):
      outputFileName = outLoc+lab+"_"+cell+"_"+fileTypeList[i]+".txt"
      outputFile = open(outputFileName,"w")
      outputFile.write("\t".join(["FACTOR","PEARSON"]+headerList)+"\n")
      for tf in tfList: outputFile.write("\t".join([tf,pearsonDict[tf]]+[e[i] for e in aucDict[tf]])+"\n")
      outputFile.close()

heFactors = [e+"\t" for e in ["ATF3", "CEBPB", "CTCF", "E2F6", "EGR1", "ELF1", "ETS1", "FOS", "FOSL1", "GABP", 
             "GATA1", "GATA2", "IRF1", "JUN", "JUND", "MAX", "MEF2A", "MYC", "NFE2", "NRF1", 
             "REST", "PU1", "SIX5", "SP1", "SP2", "SRF", "STAT1", "TAL1", "USF1", "USF2", 
             "YY1", "ZBTB33", "ZBTB7A", "ZNF263", "AR_R1881", "GR_withDEX"]]
inFileList = [outLoc+"UW_K562_10AUC.txt",  outLoc+"UW_LnCaP_10AUC.txt", outLoc+"UW_m3134_with_DEX_10AUC.txt"]
heFileName = outLoc+"HE_10AUC.txt"
os.system("grep -h -E '"+"|".join(heFactors)+"' "+" ".join(inFileList)+" | grep -h -v -E 'EFOS|EJUND' > "+heFileName)




# Import
import os
import sys
from glob import glob

# Input
rocLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM_ALLTC/"
pearsonLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MotifDinucl/table/"

# Parameters
nonFactor = ["P300","TBP","ER_0min","ER_2min","ER_5min","ER_10min","P53","P53_K562","AR_1nmR","AR_10nmR","GR_woDEX"]

# Lab Parameters
labList = ["DU","UW"]

flagFirst = True

outFileList = []

# Iterating on lab
for lab in labList:

  # Cell Parameters
  if(lab == "DU"):
    cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
    aucSelection = ["HMMD_DU_BIAS","HMMD_DU"]
  elif(lab == "UW"):
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX"]
    aucSelection = ["HMMD_UW_BIAS","HMMD_UW"]

  # Iterating on cell type
  for cell in cellList:

    # Reading pearson correlation
    pearsonFileName = pearsonLoc+lab+"_"+cell+"/pearson.txt"
    pearsonFile = open(pearsonFileName,"r")
    pearsonFile.readline()
    pearsonDict = dict()
    for line in pearsonFile:
      ll = line.strip().split("\t")
      if(ll[0] in nonFactor): continue
      pearsonDict[ll[0]] = ll[1] # 50-bp pearson
    pearsonFile.close()

    # Iterating on each TF
    tfList = sorted(pearsonDict.keys())
    aucDict = dict()
    nucDict = dict()
    nucHeaderDict = dict()
    for tf in tfList:

      # Fetching all AUC
      aucFileName = rocLoc+cell+"/"+tf+"_stats.txt"
      aucFile = open(aucFileName,"r")
      curr_aucDict = dict()
      for line in aucFile:
        ll = line.strip().split()
        if(lab == "DU"): vec = ll[2]
        if(lab == "UW"): vec = ll[2]
        #if(lab == "DU"): vec = [ll[1],ll[2],ll[5],ll[8]]
        #if(lab == "UW"): vec = [ll[1],ll[2],ll[11],ll[14]]
        curr_aucDict[ll[0]] = vec # ORDER = AUC, AUC_10, AUC_HMM, AUC_BIAS
      aucFile.close()

      # Selecting AUC vector
      aucDict[tf] = []
      for e in aucSelection: aucDict[tf].append(curr_aucDict[e])
      aucDict[tf] = str( float(aucDict[tf][0]) - float(aucDict[tf][1]) )

      # Fetching nucleotide frequencies
      pwmFileName = pearsonLoc+lab+"_"+cell+"/PWM/"+tf+"_align.pwm"
      pwmFile = open(pwmFileName,"r")
      nucRef = ["A","C","G","T"]
      nucVec = []
      for line in pwmFile: nucVec.append(sum([float(e) for e in line.strip().split(" ")]))
      pwmFile.close()

      # Mononuc freq.
      nucHeader = []
      nucRes = [] 
      summ = sum(nucVec)
      for i in range(0,len(nucVec)):
        nucHeader.append(nucRef[i])
        nucRes.append(str(round(nucVec[i]/summ,4)))

      # Dinuc freq.
      for i in range(0,len(nucVec)-1):
        for j in range(i,len(nucVec)):
          nucHeader.append(nucRef[i]+nucRef[j])
          nucRes.append(str(round((nucVec[i]+nucVec[j])/summ,4)))

      nucDict[tf] = nucRes
      nucHeaderDict[tf] = nucHeader

    # Writing file
    fileTypeList = ["10AUC"]
    for i in range(0,len(fileTypeList)):
      outputFileName = outLoc+lab+"_"+cell+".txt"
      outFileList.append(outputFileName)
      outputFile = open(outputFileName,"w")
      try:
        if(flagFirst):
          outputFile.write("\t".join(["FACTOR","PEARSON","HINT_BC-HINT"]+nucHeaderDict[tf])+"\n")
          flagFirst = False
        for tf in tfList: outputFile.write("\t".join([tf,pearsonDict[tf],aucDict[tf]]+nucDict[tf])+"\n")
      except Exception: continue
      outputFile.close()

outFileName = outLoc+"table.txt"
os.system("cat "+" ".join(outFileList)+" > "+outFileName)



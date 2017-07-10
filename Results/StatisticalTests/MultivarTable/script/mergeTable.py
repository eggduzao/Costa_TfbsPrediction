
# Import
import os
import sys

# Parameters
l1 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table1/"
l2 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table2/"
l3 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table3/"
l4 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table4/"
l5 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table5/"
l6 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table6/"
l7 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table7/"
l8 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table8/"
l9 = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table9/"
ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/table/"
inList = [ "DU_H1hesc","DU_HeLaS3","DU_HepG2","DU_Huvec","DU_K562","DU_Mcf7",
           "UW_denovo","UW_HepG2","UW_Huvec","UW_K562","UW_LnCaP","UW_m3134_with_DEX"]

# Datasets/Factors to remove
nonFactor = ["P300","TBP","ER_0min","ER_2min","ER_5min","ER_10min","P53","P53_K562","AR_1nmR","AR_10nmR","GR_woDEX"]

# Execution
for inName in inList:
  inFileVec = [e+inName+".txt" for e in [l1,l2,l3,l4,l5,l6,l7,l8,l9]]
  tempFileName = ol+inName+"_temp.txt" 
  resFileName = ol+inName+".txt"
  os.system("paste -d'\\t' "+" ".join(inFileVec)+" > "+tempFileName)
  os.system("grep -v -E '"+"|".join(nonFactor)+"' "+tempFileName+" > "+resFileName)
  os.system("rm "+tempFileName)



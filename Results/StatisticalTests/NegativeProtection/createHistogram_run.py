
# Import
import os
import sys

# Input
il = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NegativeProtection/input/"
pl = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"
ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NegativeProtection/results/"
puList = ["AR_ethl_Y","AR_R1881_Y","ER_40min_Y","ER_160min_Y","GR_withDEX_Y"]
fpList = ["LnCaP/HINT-BC_D_DU", "LnCaP/HINT-BC_D_DU", "Mcf7/HINT-BC_D_DU", "Mcf7/HINT-BC_D_DU", "m3134_with_DEX/HINT-BC_D_UW"]
labelList = ["AR_ethl", "AR_R1881", "ER_40min", "ER_160min", "GR_withDEX"]

# Creating individual histograms
outList = []
os.system("mkdir -p "+ol)
for i in range(0,len(puList)):
  label = labelList[i]
  bedFileName1 = il+puList[i]+".bed"
  bedFileName2 = pl+fpList[i]+".bed"
  outFileName = ol+labelList[i]+".txt"
  outFileNameSizeDist = ol+labelList[i]+"_size.txt"
  outFileNameStats = ol+labelList[i]+"_stats.txt"
  outList.append(outFileName)
  os.system("python /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NegativeProtection/createHistogram_pipeline.py "+" ".join([label,bedFileName1,bedFileName2,outFileName,outFileNameSizeDist,outFileNameStats]))

# Merging histograms
os.system("paste "+" ".join(outList)+" > "+ol+"histograms.txt")



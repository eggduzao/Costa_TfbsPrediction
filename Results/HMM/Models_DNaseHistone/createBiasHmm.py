###################################################################################################
# This script creates all bias HMMs based on DNase count data and histone modification
# norm/slope data.
###################################################################################################

# Import
import os
import sys

# Iterating on cell type
cellTypeList = ["H1hesc", "HeLaS3", "HepG2", "Huvec", "K562"]

# Loop on cell type
for cell in cellTypeList:

  # Iterating on histone modifications
  histoneList = ["H3K4me1", "H3K4me3"]

  # Histone loop
  for histone in histoneList:

    if(histone == "H3K4me1"): prox = "distal"
    else: prox = "proximal"
  
    # Create signal
    slopeWindowSize="9"
    perNorm="98"
    perSlope="98"
    coordFileName="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNaseHistone/"+cell+"/TrainingRegions.bed"
    signalFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/DU_"+cell+"/DNaseBiasCorrected.bw"
    outputLocation="./"
    os.system("createSignal "+slopeWindowSize+" "+perNorm+" "+perSlope+" "+coordFileName+" "+signalFileName+" "+outputLocation)

    # Train HMM
    clearNonExistingTrans="y"
    dataModeList="n,n,n,n"
    starterModel="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNaseHistone/"+cell+"/Model/blank.hmm"
    trainSet="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNaseHistone/"+cell+"/Annotation/stt/"+histone+"_"+prox+".stt"
    dl="./"
    hl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/DU_"+cell+"/"
    signalList=dl+"DNaseBiasCorrected_norm.bw,"+dl+"DNaseBiasCorrected_slope.bw,"+hl+histone+"_norm.bw,"+hl+histone+"_slope.bw"
    outputLocation="./"
    os.system("trainHMMEgg "+clearNonExistingTrans+" "+dataModeList+" "+starterModel+" "+trainSet+" "+signalList+" "+outputLocation)

    modelLoc = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNaseHistone/"
    os.system("mv ./model.hmm "+modelLoc+cell+"/Model/bias/"+histone+"_"+prox+".hmm")

# Remove generated files
os.system("rm *.bw *.wig")



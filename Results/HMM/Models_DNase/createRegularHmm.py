###################################################################################################
# This script creates all regular DNase-only HMMs.
###################################################################################################

# Import
import os
import sys

# Cell Parameters
cellTypeList = ["H1hesc", "HeLaS3", "HepG2", "Huvec", "K562"]

# Cell Loop
for cell in cellTypeList:

    # Train HMM
    clearNonExistingTrans="y"
    dataModeList="n,n"
    starterModel="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/blank.hmm"
    trainSet="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase.stt"
    dl="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/DU_"+cell+"/"
    signalList=dl+"DNase_norm.bw,"+dl+"DNase_slope.bw"
    outputLocation="/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Model/"
    os.system("trainHMMEgg "+clearNonExistingTrans+" "+dataModeList+" "+starterModel+" "+trainSet+" "+signalList+" "+outputLocation)
    os.system("mv "+outputLocation+"model.hmm "+outputLocation+"regular.hmm")



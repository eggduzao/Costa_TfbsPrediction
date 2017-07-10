
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["H1hesc", "K562"]
for cell in cellList:

  # Factor Loop
  rl = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/"+cell+"/"
  tfList = [e.split("/")[-1].split("_")[0] for e in glob(rl+"*_roc.txt")]
  for tf in tfList:

    # Parameters
    ml="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/"
    pl="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Results/"+cell+"/"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC/"+cell+"/"
    os.system("mkdir -p "+outputLocation)
    mpbsName = tf
    inList = [

# Baseline
ml+"fdr_4.bed",
pl+"FS_DU.bed",
pl+"TC_DU.bed",
pl+"Protection_DU.bed",
pl+"TCH_DU.bed",

# Competing
pl+"Boyle_DU.bed",
pl+"Centipede_90.bed",
pl+"Cuellar_90.bed",
pl+"Dnase2Tf_DU.bed",
pl+"FLR_90.bed",
pl+"Neph_DU.bed",
pl+"PIQ_90.bed",
pl+"Wellington_DU.bed",
pl+"BinDNase_80.bed",

# HINT
pl+"HINT-BC_D_DU.bed",
pl+"HINT-BCN_D_DU.bed",
pl+"HINT_D_DU.bed",
pl+"HINT_H13_DU.bed",
pl+"HINT_D13_DU.bed",
pl+"HINT-BC_D13_DU.bed"

]
    bedList = ",".join(inList)
    labelList = ",".join(["PWM"]+[e.split("/")[-1].split(".")[0] for e in inList[1:]])

    # Execution
    myL = "_".join([cell,"ROC"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 24:00 -M 48000 -S 100 -P izkf -R \"select[hpcwork]\" ./rocFromBedSBSC_pipeline.zsh "
    clusterCommand += mpbsName+" "+labelList+" "+bedList+" "+outputLocation
    os.system(clusterCommand)
    # 



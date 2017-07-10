
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["K562"]
for cell in cellList:

  # Factor Loop
  rl = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/ROC_NM3/"+cell+"/"
  tfList = [e.split("/")[-1].split("_")[0] for e in glob(rl+"*_roc.txt")]
  for tf in tfList:

    # Parameters
    ml="/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/"
    pld = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Results/"+cell+"/"
    pl="/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/ROC/"+cell+"/"
    os.system("mkdir -p "+outputLocation)
    mpbsName = tf
    inList = [
ml+"fdr_4.bed",
pld+"FS_DU.bed",
pld+"TC_DU.bed",
pld+"HINT-BC_D_DU.bed",
pld+"HINT_D_DU.bed",
pl+"ATAC_TC_50.bed",
pl+"scATAC_TC_50.bed",
pl+"ATAC_HINT_oneside_1.bed",
pl+"ATAC_HINT_oneside_3.bed",
pl+"ATAC_HINT_oneside_5.bed",
pl+"ATAC_HINT_oneside_7.bed",
pl+"ATAC_HINT_twoside_1.bed",
pl+"ATAC_HINT_twoside_3.bed",
pl+"ATAC_HINT_twoside_5.bed",
pl+"ATAC_HINT_twoside_7.bed",
pl+"sc_ATAC_HINT_twoside_1.bed",
pl+"ATAC_HINT_twoside_1_SHIFT.bed", 
pl+"ATAC_HINT_twoside_1_SHIFT_BC.bed",
pl+"sc_ATAC_HINT_twoside_1_SHIFT.bed",
pl+"sc_ATAC_HINT_twoside_1_SHIFT_BC.bed"
    ]
    bedList = ",".join(inList)
    labelList = ",".join(["PWM"]+[e.split("/")[-1].split(".")[0] for e in inList[1:]])

    # Execution
    myL = "_".join([cell,"ROC"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 10:00 -M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./rocFromBedSBSC_pipeline.zsh "
    clusterCommand += mpbsName+" "+labelList+" "+bedList+" "+outputLocation
    os.system(clusterCommand)
    # 


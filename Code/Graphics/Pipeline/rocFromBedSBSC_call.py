
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["H1hesc", "K562"]
cellList = ["K562"]
for cell in cellList:

  # Factor Loop
  rl = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_SBTC/"+cell+"/"
  tfList = [e.split("/")[-1].split("_")[0] for e in glob(rl+"*_roc.txt")]
  tfList = ["MEF2A"]
  for tf in tfList:

    # Parameters
    ml="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"+cell+"_Evidence/"
    pl="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+cell+"/"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/"+cell+"/"
    os.system("mkdir -p "+outputLocation)
    mpbsName = tf
    inList = [
ml+"fdr_4.bed",
pl+"Boyle_DU.bed",
pl+"Centipede_80.bed",
pl+"Centipede_85.bed",
pl+"Centipede_90.bed",
pl+"Centipede_95.bed",
pl+"Centipede_99.bed",
pl+"Cuellar_80.bed",
pl+"Cuellar_85.bed",
pl+"Cuellar_90.bed",
pl+"Cuellar_95.bed",
pl+"Cuellar_99.bed",
pl+"Dnase2Tf_DU.bed",
pl+"Dnase2Tf_rank.bed",
pl+"FLR_80.bed",
pl+"FLR_85.bed",
pl+"FLR_90.bed",
pl+"FLR_95.bed",
pl+"FLR_99.bed",
pl+"FS_DU.bed",
pl+"HINT-BC_D_DU.bed",
pl+"HINT-BCN_D_DU.bed",
pl+"HINT_D_DU.bed",
pl+"Neph_DU.bed",
pl+"PIQ_80.bed",
pl+"PIQ_85.bed",
pl+"PIQ_90.bed",
pl+"PIQ_95.bed",
pl+"PIQ_99.bed",
pl+"TC_DU.bed",
pl+"Wellington_DU.bed",
pl+"Wellington_rank.bed",
pl+"Protection_DU.bed",
pl+"BinDNase_80.bed",
pl+"BinDNase_85.bed",
pl+"BinDNase_90.bed",
pl+"BinDNase_95.bed",
pl+"BinDNase_99.bed",
pl+"BinDNase_rank.bed"
]
    bedList = ",".join(inList)
    typeList = ",".join(["SC" if "Cuellar" in e or "FS" in e or "TC" in e or "fdr" in e or "Protection" in e else "SB" for e in inList])
    labelList = ",".join(["PWM"]+[e.split("/")[-1].split(".")[0] for e in inList[1:]])

    # Execution
    myL = "_".join([cell,tf,"ROC"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 100:00 -M 60000 -S 100 -P izkf -R \"select[hpcwork]\" ./rocFromBedSBSC_pipeline.zsh "
    clusterCommand += mpbsName+" "+typeList+" "+labelList+" "+bedList+" "+outputLocation
    os.system(clusterCommand)
    # 


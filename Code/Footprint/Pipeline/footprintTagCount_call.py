
# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/"
inFileList = [ 

# REGULAR
il+"Mcf7/HMM_DNase_DU_ModelMcf7.bed",
il+"LnCaP/HMM_DNase_UW_ModelLnCaP.bed",
il+"m3134_with_DEX/HMM_DNase_UW_Modelm3134.bed",
il+"m3134_wo_DEX/HMM_DNase_UW_Modelm3134.bed",

# BIAS
il+"Mcf7/HMM_DNase_DU_BIAS_ModelMcf7.bed",
il+"LnCaP/HMM_DNase_UW_BIAS_ModelLnCaP.bed",
il+"m3134_with_DEX/HMM_DNase_UW_BIAS_Modelm3134.bed",
il+"m3134_wo_DEX/HMM_DNase_UW_BIAS_Modelm3134.bed",

# NAKED
il+"H1hesc/HMM_DNase_DU_NAKED_ModelH1hesc.bed",
il+"HeLaS3/HMM_DNase_DU_NAKED_ModelHeLaS3.bed",
il+"HepG2/HMM_DNase_DU_NAKED_ModelHepG2.bed",
il+"Huvec/HMM_DNase_DU_NAKED_ModelHuvec.bed",
il+"K562/HMM_DNase_DU_NAKED_ModelK562.bed",
il+"Mcf7/HMM_DNase_DU_NAKED_ModelMcf7.bed",
il+"HepG2/HMM_DNase_UW_NAKED_ModelHepG2.bed",
il+"Huvec/HMM_DNase_UW_NAKED_ModelHuvec.bed",
il+"K562/HMM_DNase_UW_NAKED_ModelK562.bed",
il+"LnCaP/HMM_DNase_UW_NAKED_ModelLnCaP.bed",
il+"m3134_with_DEX/HMM_DNase_UW_NAKED_Modelm3134.bed",
il+"m3134_wo_DEX/HMM_DNase_UW_NAKED_Modelm3134.bed"

]

# Cell Loop
for inFileName in inFileList:

  # Auxiliary
  cell = inFileName.split("/")[-2]
  lab = inFileName.split("/")[-1].split("_")[2]
  cond = inFileName.split("/")[-1].split("_")[3]
  if(cond not in ["BIAS","NAKED"]): cond = "REGULAR"

  # Parameters
  windowLen = "200"
  fpFileName = inFileName
  signalFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/DNase.bw"
  outputFileName = inFileName[:-4]+"_TC.bed"

  # Execution
  myL = "_".join([cell,lab,cond,"FTC"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 100:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./footprintTagCount_pipeline.zsh "
  clusterCommand += windowLen+" "+fpFileName+" "+signalFileName+" "+outputFileName
  os.system(clusterCommand)



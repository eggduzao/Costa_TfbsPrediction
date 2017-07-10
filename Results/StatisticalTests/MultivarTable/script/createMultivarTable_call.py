
# Import
import os
import sys

# Lab Loop
labList = ["DU", "UW"]
for lab in labList:

  # Cell Loop
  if(lab == "DU"): 
    cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
  else:
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX","m3134_wo_DEX","denovo"]
  for cell in cellList:

    # Execution
    #myL = "_".join([lab,cell,"CMT2"])
    #clusterCommand = "bsub "
    #clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    #clusterCommand += "-W 10:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./createMultivarTable_pipeline.zsh "
    #clusterCommand += lab+" "+cell
    #os.system(clusterCommand)

    clusterCommand = "python /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/MultivarTable/script/createMultivarTable9.py "+lab+" "+cell
    os.system(clusterCommand)



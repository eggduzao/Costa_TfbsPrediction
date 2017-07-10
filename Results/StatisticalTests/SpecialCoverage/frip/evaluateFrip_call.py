
# Import
import os
import sys

# Lab Loop
labList = ["DU"]
for lab in labList:

  # Cell Loop
  cellList = ["H1hesc", "K562"]
  for cell in cellList:

    # Execution
    myL = "EF_"+lab+"_"+cell
    clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand +="-W 120:00 -M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./evaluateFrip_pipeline.zsh "
    clusterCommand +=lab+" "+cell
    os.system(clusterCommand)



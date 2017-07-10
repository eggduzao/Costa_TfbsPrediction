
# Import
import os
import sys
from glob import glob

# Method Loop
methodVec = [
  "Boyle",
  "Centipede_80",
  "Centipede_85",
  "Centipede_90",
  "Centipede_95",
  "Centipede_99",
  "Cuellar_80",
  "Cuellar_85",
  "Cuellar_90",
  "Cuellar_95",
  "Cuellar_99",
  "Dnase2Tf",
  "FLR_80",
  "FLR_85",
  "FLR_90",
  "FLR_95",
  "FLR_99",
  "FS",
  "HINTBC",
  "HINTBCN",
  "HINT",
  "Neph",
  "PIQ_80",
  "PIQ_85",
  "PIQ_90",
  "PIQ_95",
  "PIQ_99",
  "Protection",
  "TC",
  "Wellington",
  "PWM"
]
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/H1hesc_GM12878/multivar_table/"
for m in methodVec:

  # Execution
  myL = "FMT_"+m
  clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 5:00 -M 24000 "
  clusterCommand += "-S 100 -R \"select[hpcwork]\" ./fetchMultivarTable_pipeline.zsh "+m+" "+outLoc
  os.system(clusterCommand)
  # -P izkf



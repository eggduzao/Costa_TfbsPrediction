
# Import
import os
import sys
from glob import glob

# Cell Loop
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Footprint Loop
  predList = [
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
    "HINT-BC_D_DU",
    "HINT-BCN_D_DU",
    "HINT_D_DU",
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
  for pred in predList:

    # TF Loop
    tfList = [e.split("/")[-1].split(".")[0] for e in glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/PWM/*.pwm")]
    for tf in tfList:

      # Parameters
      if(pred == "PWM"): mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+pred+"/"+tf+".bed"
      else: mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+cell+"/"+pred+"/"+tf+".bed"
      flrFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/FLR_batch/"+tf+".bed"
      ol = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/metrics/flr/"+cell+"/"+pred+"/"
      outputFileName = ol+tf+".bed"
      os.system("mkdir -p "+ol)

      # Execution
      os.system("intersectBed -a "+flrFileName+" -b "+mpbsFileName+" -wa -u > "+outputFileName)



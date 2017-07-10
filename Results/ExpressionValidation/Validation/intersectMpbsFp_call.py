
# Import
import os
import sys

# Cell Loop
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Footprint Loop
  predList = [
    "Boyle",
    "Centipede_80_batch",
    "Centipede_85_batch",
    "Centipede_90_batch",
    "Centipede_95_batch",
    "Centipede_99_batch",
    "Cuellar_80_batch",
    "Cuellar_85_batch",
    "Cuellar_90_batch",
    "Cuellar_95_batch",
    "Cuellar_99_batch",
    "Dnase2Tf",
    "FLR_80_batch",
    "FLR_85_batch",
    "FLR_90_batch",
    "FLR_95_batch",
    "FLR_99_batch",
    "FS_batch",
    "HINT-BC_D_DU",
    "HINT-BCN_D_DU",
    "HINT_D_DU",
    "Neph",
    "PIQ_80_batch",
    "PIQ_85_batch",
    "PIQ_90_batch",
    "PIQ_95_batch",
    "PIQ_99_batch",
    "Protection_batch",
    "TC_batch",
    "Wellington"
  ]
  for pred in predList:

    opred = pred.split("_batch")[0]

    # Parameters
    mpbsFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/MPBS/genome_mpbs.bed"
    fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"+pred+".bed"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+cell+"/"+opred+"/"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "IMF_"+cell+"_"+opred
    clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 5:00 -M 16384 -S 100 -R \"select[hpcwork]\" ./intersectMpbsFp_pipeline.zsh "
    clusterCommand += mpbsFileName+" "+fpFileName+" "+outputLocation
    os.system(clusterCommand)
    # -P izkf

# Parameters
mpbsFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/MPBS/genome_mpbs.bed"
fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/PWM_batch.bed"
outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/PWM/"
os.system("mkdir -p "+outputLocation)

# Execution
myL = "IMF_PWM"
clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
clusterCommand += "-W 5:00 -M 16384 -S 100 -R \"select[hpcwork]\" ./intersectMpbsFp_pipeline.zsh "
clusterCommand += mpbsFileName+" "+fpFileName+" "+outputLocation
os.system(clusterCommand) 
# -P izkf



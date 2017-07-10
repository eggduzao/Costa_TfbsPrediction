
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
  predList = ["PWM"]
  for pred in predList:

    # TF Loop
    if(pred == "PWM"): tfList = [e.split("/")[-1].split(".")[0] for e in glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+pred+"/*.bed")]
    else: tfList = [e.split("/")[-1].split(".")[0] for e in glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+cell+"/"+pred+"/*.bed")]
    for tf in tfList:

      # Parameters
      if(pred == "PWM"): mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+pred+"/"+tf+".bed"
      else: mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/intersection/"+cell+"/"+pred+"/"+tf+".bed"
      signalFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/DU_"+cell+"/DNaseBiasCorrected.bw"
      olt = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/metrics/tc/"+cell+"/"+pred+"/"
      olp = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/metrics/protection/"+cell+"/"+pred+"/"
      olf = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Validation/metrics/fos/"+cell+"/"+pred+"/"
      outputFileNameTc = olt+tf+".bed"
      outputFileNameProt = olp+tf+".bed"
      outputFileNameFos = olf+tf+".bed"
      os.system("mkdir -p "+olt)
      os.system("mkdir -p "+olp)
      os.system("mkdir -p "+olf)

      # Execution
      myL = "ETP_"+cell+"_"+pred+"_"+tf
      clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
      clusterCommand += "-W 10:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./evaluateTcProtFos_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+signalFileName+" "+outputFileNameTc+" "+outputFileNameProt+" "+outputFileNameFos
      os.system(clusterCommand)
      # 

  """
  # TF Loop
  tfList = [e.split("/")[-1].split(".")[0] for e in glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")]
  for tf in tfList:

    # Parameters
    mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/"+tf+".bed"
    signalFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Bias/DU_"+cell+"/DNaseBiasCorrected.bw"
    olt = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/P/TC/"+cell+"/"
    olp = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/P/PROT/"+cell+"/"
    olf = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/P/FOS/"+cell+"/"
    outputFileNameTc = olt+tf+".bed"
    outputFileNameProt = olp+tf+".bed"
    outputFileNameFos = olf+tf+".bed"
    os.system("mkdir -p "+olt)
    os.system("mkdir -p "+olp)
    os.system("mkdir -p "+olf)

    # Execution
    myL = "ETP_"+cell+"_"+tf
    clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 10:00 -M 12000 -S 100 -R \"select[hpcwork]\" ./evaluateTcProtFos_pipeline.zsh "
    clusterCommand += mpbsFileName+" "+signalFileName+" "+outputFileNameTc+" "+outputFileNameProt+" "+outputFileNameFos
    os.system(clusterCommand)
    # -P izkf
  """

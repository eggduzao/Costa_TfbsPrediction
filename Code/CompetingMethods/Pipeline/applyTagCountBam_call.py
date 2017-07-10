
# Import
import os
import sys
from glob import glob

# DNASE

# Cell Loop
cellList = ["K562"]
for cell in cellList:

  # Signal Loop
  signalList = ["ATAC","DNase"]
  for signal in signalList:

    # Window size
    windowList = ["20","50","100"]
    for w in windowList:

      # TF Loop
      tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")
      for tfFile in tfList:

        # Parameters
        tfName = tfFile.split("/")[-1].split(".")[0]
        totalWindow = w
        mpbsFileName = tfFile
        if(signal == "ATAC"):
          bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/"+cell+"/ATAC.bam"
          outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/ATAC_TC_"+w+"/"
        elif(signal == "DNase"):
          bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_DU/"+cell+"/DNase.bam"
          outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/DNase_TC_"+w+"/"
        outputFileName = outputLocation+tfName+".bed"
        os.system("mkdir -p "+outputLocation)

        # Execution
        myL = "_".join([cell,signal,"TCBAM"])
        clusterCommand = "bsub "
        clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
        clusterCommand += "-W 200:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyTagCountBam_pipeline.zsh "
        clusterCommand += totalWindow+" "+mpbsFileName+" "+bamFileName+" "+outputFileName
        #os.system(clusterCommand)

# HISTONE

# Cell Loop
cellList = ["H1hesc", "K562"]
for cell in cellList:

  # TF Loop
  tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")
  for tfFile in tfList:
  
    # Parameters
    tfName = tfFile.split("/")[-1].split(".")[0]
    totalWindow = "200"
    mpbsFileName = tfFile
    bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/Histone/"+cell+"/H3K4me3.bam"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Results/"+cell+"/Histone_TC/"
    outputFileName = outputLocation+tfName+".bed"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([cell,"HTCBAM"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 200:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyTagCountBam_pipeline.zsh "
    clusterCommand += totalWindow+" "+mpbsFileName+" "+bamFileName+" "+outputFileName
    #os.system(clusterCommand)

# Single Cell

# Cell Loop
cellList = ["K562"]
for cell in cellList:

  # Signal Loop
  signalList = ["ATAC"]
  for signal in signalList:

    # Window size
    windowList = ["50","100"]
    for w in windowList:

      # TF Loop
      tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")
      for tfFile in tfList:

        # Parameters
        tfName = tfFile.split("/")[-1].split(".")[0]
        totalWindow = w
        mpbsFileName = tfFile
        bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/sc"+cell+"/ATAC.bam"
        outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results_ATAC/Footprints/Results/"+cell+"/scATAC_TC_"+w+"/"
        outputFileName = outputLocation+tfName+".bed"
        os.system("mkdir -p "+outputLocation)

        # Execution
        myL = "_".join([cell,signal,"TCBAM"])
        clusterCommand = "bsub "
        clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
        clusterCommand += "-W 200:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyTagCountBam_pipeline.zsh "
        clusterCommand += totalWindow+" "+mpbsFileName+" "+bamFileName+" "+outputFileName
        os.system(clusterCommand)



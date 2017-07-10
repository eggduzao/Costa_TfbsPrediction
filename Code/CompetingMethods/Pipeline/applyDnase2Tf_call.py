
# Import
import os
import sys

# Lab Loop
#labList = ["DU", "UW"]
#labLocList = ["DNase", "DNaseUW"]
labList = ["DU"]
labLocList = ["DNase_DU"]
for i in range(0,len(labList)):

  lab = labList[i]
  labLoc = labLocList[i]

  # Cell Loop
  if(lab == "DU"):
    #cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
    cellList = ["GM12878"]
  else:
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX","m3134_wo_DEX","denovo"]
  for cell in cellList:
  
    # Parameters
    regionFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/"+labLoc+"/"+cell+"/DNasePeaks.bed"
    dataFilePath = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/Dnase2Tf/DNase"
    mapFilePath = "/hpcwork/izkf/projects/TfbsPrediction/Data/GersteinMappability/HG19/"
    dtfFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/Dnase2Tf/dinuc_freq_table_ac_DNase.txt"
    genomePath = "/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/"
    outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"
    outputFileName = outLoc+"Dnase2Tf.bed"
    os.system("mkdir -p "+outLoc)

    # Execution
    myL = "_".join([lab,cell,"AD2T"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 100:00 -M 30000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyDnase2Tf_pipeline.zsh "
    clusterCommand += regionFileName+" "+dataFilePath+" "+mapFilePath+" "
    clusterCommand += dtfFileName+" "+genomePath+" "+outputFileName
    os.system(clusterCommand)




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
    if("m3134" in cell): organism = "mm9"
    else: organism = "hg19"
    bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/"+labLoc+"/"+cell+"/DNase.bam"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/Dnase2Tf/"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([lab,cell,"CD2TS"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 200:00 -M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./createDnase2TfSignal_pipeline.zsh "
    clusterCommand += organism+" "+bamFileName+" "+outputLocation
    os.system(clusterCommand)




# Import
import os
import sys
from glob import glob

# Lab Loop
#labList = ["DU", "UW"]
labList = ["DU"]
for lab in labList:

  # Cell Loop
  if(lab == "DU"):
    #cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
    #cellList = ["H1hesc","K562"]
    #cellList = ["H1hesc"]
    cellList = ["K562"]
  else:
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX","m3134_wo_DEX","denovo"]
  for cell in cellList:
  

    # Factor Loop
    #factorList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"+cell+"_Evidence/fdr_4/*.bed")

    #factorList = [
    #    "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/MEF2A.bed",
    #    "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/REST.bed"
    #]
    factorList = [
        "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/REST.bed"
    ]

    for factorFileName in factorList:
 
      factorName = factorFileName.split("/")[-1].split(".")[0]
      if("m3134" in cell): organism = "mm9"
      else: organism = "hg19"

      # Parameters
      mpbsFileName = factorFileName
      bamFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNase.bam"
      genomeFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/"+organism.upper()+"/"+organism+".fa"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture/"
      outputFileName = outLoc+factorName
      os.system("mkdir -p "+outLoc)

      # Execution
      myL = "_".join([lab,cell,factorName,"CFMS"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 200:00 -M 12000 -S 100 "
      clusterCommand += "-P izkf -R \"select[hpcwork]\" ./createFootprintMixtureSignal_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+bamFileName+" "+genomeFileName+" "+outputFileName
      os.system(clusterCommand)



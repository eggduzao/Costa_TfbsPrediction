#!/bin/zsh

# Import
import os
import sys
from glob import glob

# Cell Paramters
il = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/"
inFileList = [il+"H1hesc/Dnase2Tf_DU.bed",il+"K562/Dnase2Tf_DU.bed"]

# Cell Loop
for inFileName in inFileList:

  # Auxiliary
  cell = inFileName.split("/")[-2]
  lab = inFileName.split("/")[-1].split(".")[0].split("_")[1]
  inName = inFileName.split("/")[-1].split(".")[0]

  # Parameters
  ext = "0"
  mpbsFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/TagCount_"+lab+".bed"
  predFileName = inFileName
  outputFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results_TC/"+cell+"/"+inName+"_"+ext+".bed"

  # Execution
  myL = "_".join([cell,lab,"EIMD2T"])
  clusterCommand = "bsub "
  clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 24:00 -M 12000 -S 100 "
  clusterCommand+= "-P izkf -R \"select[hpcwork]\" ./extendAndIntersectMaxScore_pipeline.zsh "
  clusterCommand += ext+" "+ext+" "+mpbsFileName+" "+predFileName+" "+outputFileName
  os.system(clusterCommand)



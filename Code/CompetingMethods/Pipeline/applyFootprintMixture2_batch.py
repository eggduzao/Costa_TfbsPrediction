
# Import
import os
import sys
from glob import glob

# Lab Loop
labList = ["DU"]
for lab in labList:

  # Cell Loop
  #cellList = ["H1hesc", "K562", "GM12878"]
  cellList = ["GM12878"]
  for cell in cellList:

    # Factor Loop
    factorList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")
    il = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/"
    factorList = [il+"CTCF.bed", il+"IRF1.bed", il+"PAX4.bed", il+"ZNF263.bed"]
    #if(cell == "H1hesc"): factorList = [il+"ZNF263.bed"]
    #else: factorList = [ ]
    for factorFileName in factorList:

      factorName = factorFileName.split("/")[-1].split(".")[0]
      organism = "hg19"

      # Parameters
      mpbsFileName = factorFileName
      peakFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNasePeaks.bed"
      cutsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture_batch/"+factorName+"_CUT.txt"
      seqFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture_batch/"+factorName+"_SEQ.fa"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+cell+"/FLR_batch/"
      outputFileName = outLoc+factorName+".bed"
      os.system("mkdir -p "+outLoc)

      # Execution
      myL = "_".join([lab,cell,factorName,"AFM2B"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt -W 120:00 "
      clusterCommand += "-M 24000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyFootprintMixture2_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+peakFileName+" "+cutsFileName+" "+seqFileName+" "+outputFileName
      os.system(clusterCommand)



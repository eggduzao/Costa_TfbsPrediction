
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
    cellList = ["H1hesc","K562"]
    #cellList = ["H1hesc"]
    #cellList = ["K562"]
  else:
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX","m3134_wo_DEX","denovo"]
  for cell in cellList:
  
    # Factor Loop
    #factorList = ["_".join(e.split("/")[-1].split("_")[:-1]) for e in glob("/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture/*_CUT.txt")]
    
    if(cell == "H1hesc"): 
      #factorList = ["ATF3", "BACH1", "BRCA1", "CEBPB", "CTCF", "EGR1", "FOSL1", "GABP", "JUN", "JUND", "MAFK", "MAX", "MYC", "NRF1", "P300", "POU5F1", "RAD21", "REST", "RFX5", "RXRA", "SIX5", "SP1", "SP2", "SP4", "SRF", "TBP", "TCF12", "USF1", "USF2", "YY1", "ZNF143" ]
      factorList = [ "REST", "MEF2A" ]

    elif(cell == "K562"):
      #factorList = ["ARID3A", "ATF1", "ATF3", "BACH1", "BHLHE40", "CCNT2", "CEBPB", "CTCF", "CTCFL", "E2F4", "E2F6", "EFOS", "EGATA", "EGR1", "EJUNB", "EJUND", "ELF1", "ELK1", "ETS1", "FOS", "FOSL1", "GABP", "GATA1", "GATA2", "IRF1", "JUN", "JUND", "MAFF", "MAFK", "MAX", "MEF2A", "MYC", "NFE2", "NFYA", "NFYB", "NR2F2", "NRF1", "PU1", "RAD21", "REST", "RFX5", "SIX5", "SMC3", "SP1", "SP2", "SRF", "STAT1", "STAT2", "STAT5A", "TAL1", "TBP", "THAP1", "TR4", "USF1", "USF2", "YY1", "ZBTB7A", "ZBTB33", "ZNF143", "ZNF263" ]
      factorList = [ "REST" ]

    for factor in factorList:

      # Parameters
      #mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"+cell+"_Evidence/fdr_4/"+factor+".bed"
      mpbsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/"+factor+".bed"

      peakFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_"+lab+"/"+cell+"/DNasePeaks.bed"
      cutsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture/"+factor+"_CUT.txt"
      seqFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/FootprintMixture/"+factor+"_SEQ.fa"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"+cell+"/FLR_"+lab+"_All/"
      outputFileName = outLoc+factor+".bed"
      os.system("mkdir -p "+outLoc)

      # Execution
      myL = "_".join([lab,cell,factor,"AFM2"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
      clusterCommand += "-W 200:00 -M 48000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyFootprintMixture2_pipeline.zsh "
      clusterCommand += mpbsFileName+" "+peakFileName+" "+cutsFileName+" "
      clusterCommand += seqFileName+" "+outputFileName
      os.system(clusterCommand)



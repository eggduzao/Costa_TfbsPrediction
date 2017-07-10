
# Import
import os
import sys
from glob import glob

# Lab Loop
#labList = ["DU", "UW"]
#labLocList = ["DNase", "DNaseUW"]
labList = ["DU"]
labLocList = ["DNase"]
for i in range(0,len(labList)):

  lab = labList[i]
  labLoc = labLocList[i]

  # Cell Loop
  if(lab == "DU"):
    #cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562","Mcf7","Saos2"]
    cellList = ["H1hesc","K562"]
  else:
    cellList = ["HepG2","Huvec","K562","LnCaP","m3134_with_DEX","m3134_wo_DEX","denovo"]
  for cell in cellList:
  
    # Factor Loop
    if(cell == "H1hesc"): 
      factorList = ["ATF3", "BACH1", "BRCA1", "CEBPB", "CTCF", "EGR1", "FOSL1", "GABP", "JUN", "JUND", 
                    "MAFK", "MAX", "MYC", "NRF1", "P300", "POU5F1", "RAD21", "REST", "RFX5", "RXRA", 
                    "SIX5", "SP1", "SP2", "SP4", "SRF", "TBP", "TCF12", "USF1", "USF2", "YY1", "ZNF143" ]
      pwmNList = [ "52", "3", "5", "6", "8", "11", "15", "17", "23", "22",
                   "25", "26", "28", "33", "34", "35", "8", "37", "38", "39",
                   "57", "40", "41", "42", "43", "48", "49", "52", "53", "54", "57" ]
    
    elif(cell == "K562"):
      factorList = ["ARID3A", "ATF1", "ATF3", "BACH1", "BHLHE40", "CCNT2", "CEBPB", "CTCF", "CTCFL", "E2F4",
                    "E2F6", "EFOS", "EGATA", "EGR1", "EJUNB", "EJUND", "ELF1", "ELK1", "ETS1", "FOS",
                    "FOSL1", "GABP", "GATA1", "GATA2", "IRF1", "JUN", "JUND", "MAFF", "MAFK", "MAX",
                    "MEF2A", "MYC", "NFE2", "NFYA", "NFYB", "NR2F2", "NRF1", "PU1", "RAD21", "REST",
                    "RFX5", "SIX5", "SMC3", "SP1", "SP2", "SRF", "STAT1", "STAT2", "STAT5A", "TAL1",
                    "TBP", "THAP1", "TR4", "USF1", "USF2", "YY1", "ZBTB7A", "ZBTB33", "ZNF143", "ZNF263" ]
      pwmNList = [ "1", "2", "7", "3", "4", "47", "6", "8", "8", "9",
                   "10", "16", "19", "11", "21", "22", "12", "13", "14", "16",
                   "15", "17", "18", "19", "20", "23", "22", "24", "25", "26",
                   "27", "28", "29", "30", "31", "32", "33", "36", "8", "37",
                   "38", "57", "8", "40", "41", "43", "44", "45", "46", "47",
                   "48", "50", "51", "52", "53", "54", "56", "55", "57", "58" ]
    
    for i in range(0,len(factorList)):
  
      # Parameters
      pwnN = pwmNList[i]
      factor = factorList[i]
      pwmLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/PIQ/"
      cutFileName="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"+lab+"_"+cell+"/PIQ/DNase.RData"
      outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/"+cell+"/PIQ_"+lab+"_7_2e6/"
      os.system("mkdir -p "+outputLocation)

      # Execution
      myL = "_".join([lab,cell,factor,"APQ"])
      clusterCommand = "bsub "
      clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
      clusterCommand += "-W 24:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./applyPIQ_pipeline.zsh "
      clusterCommand += pwnN+" "+factor+" "+pwmLoc+" "+cutFileName+" "+outputLocation
      os.system(clusterCommand)
# 



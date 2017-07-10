
# Import
import os
import sys
from glob import glob
 
# Cell Loop
#cellList = ["H1hesc", "K562", "GM12878"]
cellList = ["GM12878"]
for cell in cellList:
    
  # TF Loop
  tfList = [e.split("/")[-1].split(".")[0] for e in glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/PWM/*.pwm")]
  #tfList = ["AR", "ATOH1", "BCL6", "BHLHE40", "BRCA1", "CDX2", "CEBPB", "CREB1", "CTCF", "DUX4L1", "E2F1", "E2F3", "E2F4", "E2F6", "EBF1", "EGR1", "EGR2", "ELF1", "ELF5", "ELK4", "ERG", "ESR1", "ESR2", "ESRRA", "ETS1", "FEV", "FLII", "FOSL1", "FOSL2", "FOXA1", "FOXA2", "FOXH1", "FOXO1", "FOXO3", "FOXP1", "FOXP2", "GABPA", "GATA1", "GATA2", "GATA3", "GATA4", "GFI1B", "HINFP", "HLTF", "HNF1B", "HNF4A", "HNF4G", "HOXA5", "HOXA9", "HSF1", "INSM1", "IRF1", "JUN", "KLF1", "KLF4", "KLF5", "LHX3", "MAFB", "MAFF", "MAFK", "MAX", "MEF2C", "MEIS1", "MYB", "MYC", "MYCN", "MYOG", "NFATC2", "NFE2L2", "NFKB1", "NFYA", "NKX3-1", "NKX3-2", "NOBOX", "NR2C2", "NR2E3", "NR4A2", "NR5A2", "NRF1", "PAX5", "PRDM1", "REL", "RELA", "REST", "RFX2", "RFX5", "RUNX1", "RUNX2", "RXRA", "SOX3", "SOX5", "SOX10", "SOX17", "SP2", "SPI1", "SREBF1", "SREBF2", "SRF", "SRY", "STAT1", "STAT3", "STAT4", "TBP", "TCF7L1", "TCF7L2", "TCF12", "TFAP2A", "TFAP2C", "THAP1", "TP53", "TP63", "USF1", "YY1", "ZBTB33", "ZEB1", "ZFX", "ZNF143", "ZNF263", "ZNF354C"]
  for tf in tfList:
  
    # Parameters
    mpbsName = tf
    fimoThresh = "0.00001"
    coordFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/DU_"+cell+"/hd.bed"
    memeFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/MPBS/MEME/"+tf+".meme"
    priorFileNameList = "/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"+cell+"/Cuellar/DNase_prior.bw"
    genomeLocation = "/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/"
    outputLocation = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Cuellar_batch/"
    os.system("mkdir -p "+outputLocation)

    # Execution
    myL = "_".join([cell,tf,"ACP"])
    clusterCommand = "bsub "
    clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    clusterCommand += "-W 10:00 -M 8192 -S 100 -R \"select[hpcwork]\" ./applyCuellarHs_pipeline.zsh "
    clusterCommand += mpbsName+" "+fimoThresh+" "+coordFileName+" "+memeFileName+" "
    clusterCommand += priorFileNameList+" "+genomeLocation+" "+outputLocation
    os.system(clusterCommand)
# -P izkf



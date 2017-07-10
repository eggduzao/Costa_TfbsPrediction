
# Import
import os
import sys
from glob import glob

# Bam Parameters
dictLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/dict/"
normLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/norm/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/table/"
inList = [ "DUHS_A549", "DUHS_Cd20ro01794", "DUHS_Gm12878", "DUHS_H1hesc", "DUHS_Helas3",
           "DUHS_Helas3Ifna4h", "DUHS_Hepg2", "DUHS_Huvec", "DUHS_Imr90", "DUHS_K562",
           "DUHS_K562G1phase", "DUHS_K562G2mphase", "DUHS_K562Nabut", "DUHS_K562Saha1u72hr",
           "DUHS_K562Sahactrl", "DUHS_Mcf7", "DUHS_Mcf7Ctcfshrna", "DUHS_Mcf7Hypoxlac",
           "DUHS_Mcf7Randshrna", "DUHS_Monocd14", "DUHS_Sknsh",
           "UWDGF_A549", "UWDGF_Cd20ro01778", "UWDGF_Hepg2", "UWDGF_Huvec", "UWDGF_K562", 
           "UWDGF_K562Znfa41c6", "UWDGF_K562Znfp5", "UWDGF_Lhcnm2", "UWDGF_Lhcnm2Diff4d", 
           "UWHS_A549", "UWHS_Cd20ro01778", "UWHS_Gm12878", "UWHS_H1hesc", "UWHS_Helas3", 
           "UWHS_Hepg2", "UWHS_Huvec", "UWHS_K562", "UWHS_K562Znf2c10c5", 
           "UWHS_K562Znf4c50c4", "UWHS_K562Znf4g7d3", "UWHS_K562Znfa41c6", "UWHS_K562Znfb34a8", 
           "UWHS_K562Znfe103c6", "UWHS_K562Znff41b2", "UWHS_K562Znfg54a11", "UWHS_K562Znfp5", 
           "UWHS_Lhcnm2", "UWHS_Lhcnm2Diff4d", "UWHS_Mcf7", "UWHS_Mcf7Est100nm1h", 
           "UWHS_Mcf7Estctrl0h", "UWHS_Monocd14", "UWHS_Monocd14ro1746",
           "DUHS_H7hesc", "DUHS_NakedK562", "DUHS_NakedMcf7", "DUHS_LNCaP",
           "UWDGF_H7hesc", "UWDGF_NakedIMR90", "UWDGF_m3134" ]

inList = [ "DUHS_K562Sahactrl", "UWDGF_Hepg2", "UWHS_Hepg2", "UWDGF_K562"]

# Bam Loop
for inFile in inList:

  # Output Type Parameters
  typeList = ["hs"]

  # Output Type Loop
  for t in typeList:

    # Strand Parameters
    sList = ["F", "R", ""]

    # Strand Loop
    for s in sList:

      # Parameters
      inObs = dictLoc+t+"_"+inFile+"_6_obsDict"+s+".p"
      inExp = dictLoc+t+"_"+inFile+"_6_expDict"+s+".p"
      normFileName = normLoc+inFile+".txt"
      outputLocation = outLoc

      # Execution
      os.system("./createKmerTable_pipeline.zsh "+inObs+" "+inExp+" "+normFileName+" "+outputLocation)



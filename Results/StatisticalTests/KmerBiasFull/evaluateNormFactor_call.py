
# Import
import os
import sys
from glob import glob

# Bam Parameters
loc1 = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/DUHS/"
loc2 = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/UWDGF/"
loc3 = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/UWHS/"
outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBiasFull/bias/norm/"
inList = [ loc1+"A549", loc1+"Cd20ro01794", loc1+"Gm12878", loc1+"H1hesc", loc1+"Helas3",
           loc1+"Helas3Ifna4h", loc1+"Hepg2", loc1+"Huvec", loc1+"Imr90", loc1+"K562",
           loc1+"K562G1phase", loc1+"K562G2mphase", loc1+"K562Nabut", loc1+"K562Saha1u72hr",
           loc1+"K562Sahactrl", loc1+"Mcf7", loc1+"Mcf7Ctcfshrna", loc1+"Mcf7Hypoxlac",
           loc1+"Mcf7Randshrna", loc1+"Monocd14", loc1+"Sknsh",

           loc2+"A549", loc2+"Cd20ro01778", loc2+"Hepg2", loc2+"Huvec", loc2+"K562", 
           loc2+"K562Znfa41c6", loc2+"K562Znfp5", loc2+"Lhcnm2", loc2+"Lhcnm2Diff4d", 

           loc3+"A549", loc3+"Cd20ro01778", loc3+"Gm12878", loc3+"H1hesc", loc3+"Helas3", 
           loc3+"Hepg2", loc3+"Huvec", loc3+"K562", loc3+"K562Znf2c10c5", 
           loc3+"K562Znf4c50c4", loc3+"K562Znf4g7d3", loc3+"K562Znfa41c6", loc3+"K562Znfb34a8", 
           loc3+"K562Znfe103c6", loc3+"K562Znff41b2", loc3+"K562Znfg54a11", loc3+"K562Znfp5", 
           loc3+"Lhcnm2", loc3+"Lhcnm2Diff4d", loc3+"Mcf7", loc3+"Mcf7Est100nm1h", 
           loc3+"Mcf7Estctrl0h", loc3+"Monocd14", loc3+"Monocd14ro1746",

           loc1+"H7hesc", loc1+"NakedK562", loc1+"NakedMcf7", loc1+"LNCaP",

           loc2+"H7hesc", loc2+"NakedIMR90", loc2+"m3134" ]

inList = [ loc2+"Hepg2", loc2+"K562", loc3+"Hepg2" ]

# Bam Loop
for inFile in inList:

  # Initialization
  lab = inFile.split("/")[-2]
  cell = inFile.split("/")[-1]

  # Parameters
  bamFileName = inFile+".bam"
  hsFileName = inFile+".bed"
  outputFileName = outLoc+"_".join([lab,cell])+".txt"

  # Execution
  myL = "ENF_"+"_".join([lab,cell])
  clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
  clusterCommand += "-W 10:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./evaluateNormFactor_pipeline.zsh "
  clusterCommand += bamFileName+" "+hsFileName+" "+outputFileName
  os.system(clusterCommand)



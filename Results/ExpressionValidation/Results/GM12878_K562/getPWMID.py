
import os
import sys
from glob import glob

tfDict = dict()
inList = glob("/home/eg474423/rgtdata/motifs/jaspar_vertebrates/*.pwm")
for inf in inList: tfDict[inf.split("/")[-1].split(".")[2].upper()] = ".".join(inf.split("/")[-1].split(".")[:2])

tfDict["FLII"] = "MA0475.1"
tfDict["TCF7L1"] = "MA0522.1"

inFileName = "./TF_Summary_GM12878_K562.xls"
outFileName = "./PWM_ID.txt"
inFile = open(inFileName,"r")
outFile = open(outFileName,"w")
inFile.readline()
for line in inFile:
  factor = line.strip().split(",")[0]
  outFile.write(tfDict[factor]+"\n")
  #except Exception:
  #  outFile.write("NA\n")
  #  print factor
inFile.close()
outFile.close()



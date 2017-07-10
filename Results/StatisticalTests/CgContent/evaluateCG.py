
# Import
import os
import sys
from glob import glob

# Input
pwmLoc = "/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/All/"
nrLoc = "/hpcwork/izkf/projects/TfbsPrediction/Data/PWM/NuclearReceptors/"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/pwmcg.txt"
orderFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/CgContent/order.txt"
nrList = [nrLoc+"MA0007.2.AR.pwm",nrLoc+"MA0112.2.ESR1.pwm",nrLoc+"MA0113.2.NR3C1.pwm"]

# Reading order
orderVec = []
inF = open(orderFileName,"r")
for line in inF: orderVec.append(line.strip().split("/")[-1])
inF.close()

# PWM Loop
pwmcgres = dict()
for pwmFileName in glob(pwmLoc+"*.pfm")+nrList:

  if("MA0007.2.AR" in pwmFileName): pwmName = "AR"
  elif("MA0112.2.ESR1" in pwmFileName): pwmName = "ER"
  elif("MA0113.2.NR3C1" in pwmFileName): pwmName = "GR"
  else: pwmName = pwmFileName.strip().split("/")[-1].split("_")[0]
  
  pwmFile = open(pwmFileName,"r")
  sumVec = []
  for line in pwmFile:
    ll = line.strip().split(" ")
    sumVec.append(sum([float(e) for e in ll]))
  pwmFile.close()

  pwmcgres[pwmName] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "USF1"): pwmcgres["ATF3"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "CTCF"): pwmcgres["RAD21"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "ZNF143"): pwmcgres["SIX5"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "TAL1"): pwmcgres["CCNT2"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "CTCF"): pwmcgres["CTCFL"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "FOS"): pwmcgres["EFOS"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "GATA2"): pwmcgres["EGATA"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "JUNB"): pwmcgres["EJUNB"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "JUND"): pwmcgres["EJUND"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "CTCF"): pwmcgres["SMC3"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))

  if(pwmName == "AR"): pwmcgres["AR_R1881"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "AR"): pwmcgres["AR_ethl"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))

  if(pwmName == "ER"): pwmcgres["ER_40min"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))
  if(pwmName == "ER"): pwmcgres["ER_160min"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))

  if(pwmName == "GR"): pwmcgres["GR_withDEX"] = str((sumVec[1]+sumVec[2])/(sum(sumVec)))

# Bias Loop



outFile = open(outFileName,"w")
for factor in orderVec:
  outFile.write("\t".join([factor,pwmcgres[factor]])+"\n")
outFile.close()



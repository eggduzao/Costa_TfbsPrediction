import os
import sys
from glob import glob

# Parameters
n = 2563
pwmFlag = False
tcFlag = True
fosFlag = True
protFlag = True
pl = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"

# Cell Loop
cellList = ["H1hesc", "K562", "GM12878"]
tfList = glob("/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/fdr_4_inside_HS/*.bed")

# PWM
if(pwmFlag):
  toRemove = []
  for tfFile in tfList:
    inFileName = tfFile
    outFileTemp1 = inFileName+"T1"
    outFileTemp2 = inFileName+"T2"
    toRemove.append(outFileTemp2)
    os.system("sort -k5,5gr "+inFileName+" > "+outFileTemp1)
    os.system("head -n "+str(n)+" "+outFileTemp1+" > "+outFileTemp2)
    os.system("rm "+outFileTemp1)
  outFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/PWM_batch.bed"
  os.system("cat "+" ".join(toRemove)+" > "+outFileName)
  for e in toRemove: os.system("rm "+e)

# TC
if(tcFlag):
  for cell in cellList:
    toRemove = []   
    for tfFile in tfList:
      tfName = tfFile.split("/")[-1].split(".")[0]
      inFileName = pl+cell+"/TC_batch/"+tfName+".bed"
      outFileTemp1 = inFileName+"T1"
      outFileTemp2 = inFileName+"T2"
      toRemove.append(outFileTemp2)
      os.system("sort -k5,5gr "+inFileName+" > "+outFileTemp1)
      os.system("head -n "+str(n)+" "+outFileTemp1+" > "+outFileTemp2)
      os.system("rm "+outFileTemp1)
    outFileName = pl+cell+"/TC_batch.bed"
    os.system("cat "+" ".join(toRemove)+" > "+outFileName)
    for e in toRemove: os.system("rm "+e)

# FOS
if(fosFlag):
  for cell in cellList:
    toRemove = []   
    for tfFile in tfList:
      tfName = tfFile.split("/")[-1].split(".")[0]
      inFileName = pl+cell+"/FS_batch/"+tfName+".bed"
      outFileTemp1 = inFileName+"T1"
      outFileTemp2 = inFileName+"T2"
      toRemove.append(outFileTemp2)
      os.system("sort -k5,5gr "+inFileName+" > "+outFileTemp1)
      os.system("head -n "+str(n)+" "+outFileTemp1+" > "+outFileTemp2)
      os.system("rm "+outFileTemp1)
    outFileName = pl+cell+"/FS_batch.bed"
    os.system("cat "+" ".join(toRemove)+" > "+outFileName)
    for e in toRemove: os.system("rm "+e)

# PROT
if(protFlag):
  for cell in cellList:
    toRemove = []   
    for tfFile in tfList:
      tfName = tfFile.split("/")[-1].split(".")[0]
      inFileName = pl+cell+"/Protection_batch/"+tfName+".bed"
      outFileTemp1 = inFileName+"T1"
      outFileTemp2 = inFileName+"T2"
      toRemove.append(outFileTemp2)
      os.system("sort -k5,5gr "+inFileName+" > "+outFileTemp1)
      os.system("head -n "+str(n)+" "+outFileTemp1+" > "+outFileTemp2)
      os.system("rm "+outFileTemp1)
    outFileName = pl+cell+"/Protection_batch.bed"
    os.system("cat "+" ".join(toRemove)+" > "+outFileName)
    for e in toRemove: os.system("rm "+e)



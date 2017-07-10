
import os
import sys
from glob import glob

inMpbsLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/MPBSAWG/K562_Evidence/fdr_4/"
fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/K562/HINT_D_DU.bed"
#fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/K562/HINT-BC_D_DU.bed"
#fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/K562/HINT-BCN_D_DU.bed"
#fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/K562/H-HMM_D13_DU.bed"
#fpFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results_DNase/Footprints/Predictions/K562/HINT-BC_D13_DU.bed"
oLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/ContingencyProfiling/contingency_files/"

for mpbsFileName in glob(inMpbsLoc+"*.*"):

  tf = mpbsFileName.split("/")[-1].split(".")[0]

  tempy = "./tempy.bed"
  tempn = "./tempn.bed"
  os.system("grep :Y "+mpbsFileName+" | cut -f 1,2,3 > "+tempy)
  os.system("grep :N "+mpbsFileName+" | cut -f 1,2,3 > "+tempn)

  outTPT = "./TPT.bed"
  outFPT = "./FPT.bed"
  outTNT = "./TNT.bed"
  outFNT = "./FNT.bed"
  os.system("intersectBed -a "+tempy+" -b "+fpFileName+" -wa -u > "+outTPT)
  os.system("intersectBed -a "+tempy+" -b "+fpFileName+" -wa -v > "+outFNT)
  os.system("intersectBed -a "+tempn+" -b "+fpFileName+" -wa -u > "+outFPT)
  os.system("intersectBed -a "+tempn+" -b "+fpFileName+" -wa -v > "+outTNT)

  outTP = oLoc+tf+"_TP.bed"
  outFP = oLoc+tf+"_FP.bed"
  outTN = oLoc+tf+"_TN.bed"
  outFN = oLoc+tf+"_FN.bed"
  os.system("head -1000 "+outTPT+" > "+outTP)
  os.system("head -1000 "+outFPT+" > "+outFP)
  os.system("head -1000 "+outTNT+" > "+outTN)
  os.system("head -1000 "+outFNT+" > "+outFN)

  toRemove = [tempy,tempn,outTPT,outFPT,outFNT,outTNT]
  for e in toRemove: os.system("rm "+e)



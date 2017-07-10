
###################################################################################################
# IMPORT
###################################################################################################

import os
import sys
import math
from math import log
from Bio import motifs
import subprocess
import traceback
from scipy.stats.stats import pearsonr
from bx.bbi.bigwig_file import BigWigFile
sys.path.append("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code")
from util import *

###################################################################################################
# INPUT
###################################################################################################

# Input
lab = sys.argv[1]
cell = sys.argv[2]

# Input files
wl = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/"
hl = "/hpcwork/izkf/projects/TfbsPrediction/"
pearsonLoc = wl+"StatisticalTests/TFSpecificBias/Pearson/"
aucLoc = hl+"Results/Footprints/ROC_NM/"
peakLoc = hl+"Results/TFBSAWG/"
mpbsLoc = hl+"Results/MPBSAWG/"
dnaseLocUnc = hl+"Results/Signals/Counts/"
dnaseLocCorr = hl+"Results/Signals/Bias/"
dhsLoc = hl+"Data/"
fpLoc = wl+"StatisticalTests/MultivarTable/predictions/"
outputFileName = wl+"StatisticalTests/MultivarTable/table2/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table2/"+lab+"_"+cell+"/"
cFileName = wl+"StatisticalTests/MultivarTable/script/coverage.txt"
mFileName = wl+"StatisticalTests/MultivarTable/script/minvalues.txt"
os.system("mkdir -p "+tempLoc)

# Parameters
peakExtHalf = 50
motifExt = 20

###################################################################################################
# Functions
###################################################################################################

# File length function
def file_len(fname, grep=None):
  if(grep): p = subprocess.Popen("grep "+grep+" "+fname+" | wc -l", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  else: p = subprocess.Popen("wc -l "+fname, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0: raise IOError(err)
  return int(result.strip().split()[0])

# Evaluate TC FOS PROT function
def eval_tc_fos_prot(bw, mpbsFileName, rpmDict, minDict, halfWindow = 100):

  # Initialization
  mpbsFile = open(mpbsFileName, "r")
  stc = 0.0; sfs = 0.0; spr = 0.0; counter = 0.0
  minV = minDict[lab+"_"+cell]
  rpmV = rpmDict[lab+"_"+cell]

  # Iterating on mpbs file
  for line in mpbsFile:

    # Fetching signal
    ll = line.strip().split("\t")
    mLen = int(ll[2]) - int(ll[1])
    mid = (int(ll[1])+int(ll[2]))/2
    p1 = max(mid - halfWindow,0)
    p2 = mid + halfWindow
    try: signal = [min(max(e,-1000),1000) for e in aux.correctBW(bw.get(ll[0],p1,p2),p1,p2)]
    except Exception:
      print "Could not fetch signal in "+mpbsFileName+" location = "+ll[0]+":"+"-".join([str(p1),str(p2)])
      continue
    counter += 1
    
    # Summing min value to signal
    stdzSignal = [e+minV for e in signal]
    
    # Evaluating TC
    stc += sum(stdzSignal)
    
    # Evaluating FOS
    signalHalfLen = len(stdzSignal)/2
    motifHalfLen = mLen/2
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen:signalHalfLen+motifHalfLen+motifExt])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt:signalHalfLen-motifHalfLen])
    sfs += -(((nc+1)/(nr+1)) + ((nc+1)/(nl+1)))

    # Evaluating PROT
    signalHalfLen = len(signal)/2
    motifHalfLen = motifExt/2
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen:signalHalfLen+motifHalfLen+motifExt])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt:signalHalfLen-motifHalfLen])
    spr += ((nr+nl)/2) - nc

  # Evaluating averages
  if(counter > 0):
    tcmean = (stc/counter) * rpmV
    fosmean = sfs/counter
    protmean = spr/counter
    return [str(e) for e in [tcmean,fosmean,protmean]]
  else:
    print "Counter was <= 0 in "+mpbsFileName
    return ["NA", "NA", "NA"]

def evaluate_entropy(pwm):
  resVec = []
  for j in range(0,len(pwm[0])):
    entsum = 0.0
    for i in range(0,len(pwm)):
      if(pwm[i][j] == 0): continue
      else: entsum += pwm[i][j] * log(pwm[i][j],2)
    resVec.append(2-(-entsum))
  return resVec

###################################################################################################
# Pearson Bias estimate
###################################################################################################

# Fetching Pearson bias estimate
pearsonDict = dict()
pearsonFileName = pearsonLoc+lab+"_"+cell+"/pearson.txt"
pearsonFile = open(pearsonFileName,"r")
pearsonFile.readline()
for line in pearsonFile:
  ll = line.strip().split("\t") 
  pearsonDict[ll[0]] = ll[1]
pearsonFile.close()

# Factor List
factorList = sorted(pearsonDict.keys())
  
###################################################################################################
# TC, FOS, PROT
###################################################################################################

# Creating coverage dictionary
rpmDict = dict()
cFile = open(cFileName,"r")
for line in cFile:
  ll = line.strip().split("\t")
  rpmDict[ll[0]] = 1000000.0 / float(ll[1])
cFile.close()

# Creating minvalues dictionary

minDictCorr = dict()
mFile = open(mFileName,"r")
for line in mFile:
  ll = line.strip().split("\t")
  minDictCorr[ll[0]] = abs(float(ll[1]))
mFile.close()

# Initialization of files
if(cell == "Saos2"):
  dnaseFileNameUnc = dnaseLocUnc+"DU_K562/DNase.bw"
  dnaseFileNameCorr = dnaseLocCorr+"DU_K562/DNaseBiasCorrected.bw"
elif(cell == "denovo"):
  dnaseFileNameUnc = dnaseLocUnc+"UW_H7hesc/DNase.bw"
  dnaseFileNameCorr = dnaseLocCorr+"UW_H7hesc/DNaseBiasCorrected.bw"
else:
  dnaseFileNameUnc = dnaseLocUnc+lab+"_"+cell+"/DNase.bw"
  dnaseFileNameCorr = dnaseLocCorr+lab+"_"+cell+"/DNaseBiasCorrected.bw"
dnaseFileUnc = open(dnaseFileNameUnc,"r")
dnaseFileCorr = open(dnaseFileNameCorr,"r")
bwUnc = BigWigFile(dnaseFileUnc)
bwCorr = BigWigFile(dnaseFileCorr)

# MPBSs that overlap ChIP-seq
chipTcDictCorr = dict()
for factor in factorList:

  if(cell == "denovo"):
    chipTcDictCorr[factor] = ["NA", "NA", "NA"]
    continue

  # Initializing MPBS file
  if("m3134" in cell): mpbsFileName = mpbsLoc+"m3134_Evidence/fdr_4/"+factor+".bed"
  else: mpbsFileName = mpbsLoc+cell+"_Evidence/fdr_4/"+factor+".bed"
  mpbsFileNameTemp = tempLoc+factor+"_mpbs.bed"
  os.system("grep :Y "+mpbsFileName+" > "+mpbsFileNameTemp)
  
  # Corrected TC, FOS, PROT
  try: chipTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTemp,rpmDict,minDictCorr)
  except Exception:
    print "Error in CHIP bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    chipTcDictCorr[factor] = ["NA", "NA", "NA"]

  # Removing files
  os.system("rm "+mpbsFileNameTemp)

# MPBSs that overlap DNase-seq
dnaseTcDictCorr = dict()
for factor in factorList:

  # Initializing MPBS file
  if("m3134" in cell): mpbsFileName = mpbsLoc+"m3134_Evidence/fdr_4/"+factor+".bed"
  else: mpbsFileName = mpbsLoc+cell+"_Evidence/fdr_4/"+factor+".bed"
  mpbsFileNameTempD = tempLoc+factor+"_mpbs_dnase.bed"

  # Initializing DNase peaks file
  if(lab == "DU"): dhsFileName = dhsLoc+"DNase/"+cell+"/DNasePeaks.bed"
  else: dhsFileName = dhsLoc+"DNaseUW/"+cell+"/DNasePeaks.bed"
  if(cell == "denovo"): dhsFileName = dhsLoc+"DNaseUW/H7hesc/DNasePeaks.bed"
  tempDhsSortFileName = tempLoc+factor+"_dnase_sort.bed"
  os.system("sort -k1,1 -k2,2n "+dhsFileName+" > "+tempDhsSortFileName)
  os.system("intersectBed -wa -a "+mpbsFileName+" -b "+tempDhsSortFileName+" > "+mpbsFileNameTempD)
  to_remove = [mpbsFileNameTempD,tempDhsSortFileName]
  
  # Corrected
  try: dnaseTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTempD,rpmDict,minDictCorr)
  except Exception:
    print "Error in DNASE bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    dnaseTcDictCorr[factor] = ["NA", "NA", "NA"]

  # Removing files
  for e in to_remove: os.system("rm "+e)

# MPBSs that overlap Footprints
fpTcDictCorr = dict()
for factor in factorList:

  if(cell == "denovo"):
    fpTcDictCorr[factor] = ["NA", "NA", "NA"]
    continue

  # Initializing MPBS file
  if("m3134" in cell): mpbsFileName = mpbsLoc+"m3134_Evidence/fdr_4/"+factor+".bed"
  else: mpbsFileName = mpbsLoc+cell+"_Evidence/fdr_4/"+factor+".bed"
  mpbsFileNameTempF = tempLoc+factor+"_mpbs_fp.bed"

  # Initializing DNase peaks file
  fpFileName = fpLoc+lab+"_"+cell+".bed"
  tempFpExtFileName = tempLoc+factor+"_fp_ext.bed"
  fpFile = open(fpFileName,"r")
  tempFpExtFile = open(tempFpExtFileName,"w")
  for line in fpFile:
    ll = line.strip().split("\t")
    tempFpExtFile.write("\t".join([ll[0],str(int(ll[1])-5),str(int(ll[2])+5)])+"\n")
  fpFile.close()
  tempFpExtFile.close()
  tempFpSortFileName = tempLoc+factor+"_fp_sort.bed"
  os.system("sort -k1,1 -k2,2n "+tempFpExtFileName+" > "+tempFpSortFileName)
  os.system("intersectBed -wa -a "+mpbsFileName+" -b "+tempFpSortFileName+" > "+mpbsFileNameTempF)
  to_remove = [mpbsFileNameTempF,tempFpSortFileName]
  
  # Corrected
  try: fpTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTempF,rpmDict,minDictCorr)
  except Exception:
    print "Error in DNASE bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    fpTcDictCorr[factor] = ["NA", "NA", "NA"]

  # Removing files
  for e in to_remove: os.system("rm "+e)

# Closing dnase files
dnaseFileUnc.close()
dnaseFileCorr.close()

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join([
          "MINSUM_TC_AVG_CHIP_CORRECTED", "MINSUM_FOS_AVG_CHIP_CORRECTED", "MINSUM_PROT_AVG_CHIP_CORRECTED",
          "MINSUM_TC_AVG_DNASE_CORRECTED", "MINSUM_FOS_AVG_DNASE_CORRECTED", "MINSUM_PROT_AVG_DNASE_CORRECTED",
          "MINSUM_TC_AVG_FP_CORRECTED", "MINSUM_FOS_AVG_FP_CORRECTED", "MINSUM_PROT_AVG_FP_CORRECTED"
          ])+"\n")
for factor in factorList:
  outputFile.write("\t".join([str(e) for e in [
                   chipTcDictCorr[factor][0],chipTcDictCorr[factor][1],chipTcDictCorr[factor][2],
                   dnaseTcDictCorr[factor][0],dnaseTcDictCorr[factor][1],dnaseTcDictCorr[factor][2],
                   fpTcDictCorr[factor][0],fpTcDictCorr[factor][1],fpTcDictCorr[factor][2]
                   ]])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)



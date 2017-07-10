
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
outputFileName = wl+"StatisticalTests/MultivarTable/table1/"+lab+"_"+cell+".txt"
tempLoc = wl+"StatisticalTests/MultivarTable/table1/"+lab+"_"+cell+"/"
cFileName = wl+"StatisticalTests/MultivarTable/script/coverage.txt"
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

# Standardize vector function
def standardize(vec):
  minV = 999999; maxV = -999999
  for e in vec:
    if(e > maxV): maxV = e
    if(e < minV): minV = e
  try: return [(e-minV)/(maxV-minV) for e in vec]
  except Exception: return vec

# Evaluate TC FOS PROT function
def eval_tc_fos_prot(bw, mpbsFileName, rpmDict, halfWindow = 100):

  # Initialization
  mpbsFile = open(mpbsFileName, "r")
  stc = 0.0; sfs = 0.0; sfs5 = 0.0; sfs10 = 0.0; spr = 0.0; counter = 0.0
  

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

    # Evaluating TC
    stc += sum(signal)

    # Standardizing signal for FOS and PROT
    stdzSignal = standardize(signal)

    # Evaluating FOS
    signalHalfLen = len(stdzSignal)/2
    motifHalfLen = mLen/2
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen:signalHalfLen+motifHalfLen+motifExt])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt:signalHalfLen-motifHalfLen])
    sfs += -(((nc+1)/(nr+1)) + ((nc+1)/(nl+1)))

    # Evaluating FOS with gap = 5
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen+5:signalHalfLen+motifHalfLen+motifExt+5])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt-5:signalHalfLen-motifHalfLen-5])
    sfs5 += -(((nc+1)/(nr+1)) + ((nc+1)/(nl+1)))

    # Evaluating FOS with gap = 10
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen+10:signalHalfLen+motifHalfLen+motifExt+10])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt-10:signalHalfLen-motifHalfLen-10])
    sfs10 += -(((nc+1)/(nr+1)) + ((nc+1)/(nl+1)))

    # Evaluating PROT
    signalHalfLen = len(signal)/2
    motifHalfLen = motifExt/2
    nc = sum(stdzSignal[signalHalfLen-motifHalfLen:signalHalfLen+motifHalfLen])
    nr = sum(stdzSignal[signalHalfLen+motifHalfLen:signalHalfLen+motifHalfLen+motifExt])
    nl = sum(stdzSignal[signalHalfLen-motifHalfLen-motifExt:signalHalfLen-motifHalfLen])
    spr += ((nr+nl)/2) - nc

  # Evaluating averages
  if(counter > 0):
    tcmean = stc/counter
    tcnmean = tcmean*rpmDict[lab+"_"+cell]
    fosmean = sfs/counter
    fos5mean = sfs5/counter
    fos10mean = sfs10/counter
    protmean = spr/counter
    return [str(e) for e in [tcmean,fosmean,protmean,tcnmean,fos5mean,fos10mean]]
  else:
    print "Counter was <= 0 in "+mpbsFileName
    return ["NA", "NA", "NA", "NA", "NA", "NA"]

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
# AUC
###################################################################################################

# Fetching AUC HINT(BC)
aucDictAll = dict()
aucDict10 = dict()
metVec = ["PWM","TC_"+lab,"FOS_"+lab,"HMMD_"+lab,"HMMD_"+lab+"_BIAS","HMMD_"+lab+"_NAKED"]
for factor in factorList:
  aucFileName = aucLoc+cell+"/"+factor+"_stats.txt"
  try: aucFile = open(aucFileName,"r")
  except Exception: 
    aucDictAll[factor] = ["NA" for e in metVec]
    aucDict10[factor] = ["NA" for e in metVec]
    continue
  aucFile.readline()
  aucDict = dict()
  for line in aucFile:
    ll = line.strip().split("\t")
    aucDict[ll[0]] = [ll[1],ll[2]]
  aucDictAll[factor] = [aucDict[e][0] for e in metVec]
  aucDict10[factor] = [aucDict[e][1] for e in metVec]
  aucFile.close()	

###################################################################################################
# AUPR
###################################################################################################

# TODO

###################################################################################################
# MPBS/TFBS statistics
###################################################################################################

# Fetch motif+chip statistics
statsDict = dict()
for factor in factorList:
  if("m3134" in cell):
    mpbsFileName = mpbsLoc+"m3134_Evidence/fdr_4/"+factor+".bed"
    peakFileName = peakLoc+"m3134/"+factor+".bed"
  else: 
    mpbsFileName = mpbsLoc+cell+"_Evidence/fdr_4/"+factor+".bed"
    peakFileName = peakLoc+cell+"/"+factor+".bed"
  nbMotifs = file_len(mpbsFileName, grep=None)
  try: nbPeaks = file_len(peakFileName, grep=None)
  except Exception:
    statsDict[factor] = [nbMotifs,"NA","NA","NA"]
    continue
  nbMotifwithPeaks = file_len(mpbsFileName, grep=":Y")
  tempFileName = tempLoc+factor+"_temp.bed"
  peakTempFileName = tempLoc+factor+"_peak_temp.bed"
  peakTempSortFileName = tempLoc+factor+"_peak_temp_sort.bed"
  mpbsTempSortFileName = tempLoc+factor+"_mpbs_temp_sort.bed"
  to_remove = [tempFileName,peakTempFileName,peakTempSortFileName,mpbsTempSortFileName]
  peakFile = open(peakFileName,"r")
  peakTempFile = open(peakTempFileName,"w")
  for line in peakFile:
    ll = line.strip().split("\t")
    mid = (int(ll[1])+int(ll[2])) / 2
    peakTempFile.write("\t".join([ll[0],str(max(mid-peakExtHalf,0)),str(mid+peakExtHalf)])+"\n")
  peakFile.close()
  peakTempFile.close()
  os.system("cut -f 1,2,3 "+peakTempFileName+" | sort -k1,1 -k2,2n > "+peakTempSortFileName)
  os.system("cut -f 1,2,3 "+mpbsFileName+" | sort -k1,1 -k2,2n > "+mpbsTempSortFileName)
  os.system("intersectBed -u -wa -a "+peakTempSortFileName+" -b "+mpbsTempSortFileName+" > "+tempFileName)
  nbPeakswithMotif = file_len(tempFileName, grep=None)
  statsDict[factor] = [nbMotifs,nbPeaks,nbPeakswithMotif,round(100.0*float(nbPeakswithMotif)/nbPeaks,6)]
  for e in to_remove: os.system("rm "+e)
  
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
chipTcDictUnc = dict()
chipTcDictCorr = dict()
for factor in factorList:

  if(cell == "denovo"):
    chipTcDictUnc[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
    chipTcDictCorr[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
    continue

  # Initializing MPBS file
  if("m3134" in cell): mpbsFileName = mpbsLoc+"m3134_Evidence/fdr_4/"+factor+".bed"
  else: mpbsFileName = mpbsLoc+cell+"_Evidence/fdr_4/"+factor+".bed"
  mpbsFileNameTemp = tempLoc+factor+"_mpbs.bed"
  os.system("grep :Y "+mpbsFileName+" > "+mpbsFileNameTemp)

  # Uncorrected TC, FOS, PROT
  try: chipTcDictUnc[factor] = eval_tc_fos_prot(bwUnc,mpbsFileNameTemp,rpmDict)
  except Exception:
    print "Error in CHIP bwUnc eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    chipTcDictUnc[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
  
  # Corrected TC, FOS, PROT
  try: chipTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTemp,rpmDict)
  except Exception:
    print "Error in CHIP bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    chipTcDictCorr[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]

  # Removing files
  os.system("rm "+mpbsFileNameTemp)

# MPBSs that overlap DNase-seq
dnaseTcDictUnc = dict()
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

  # Uncorrected TC, FOS, PROT
  try: dnaseTcDictUnc[factor] = eval_tc_fos_prot(bwUnc,mpbsFileNameTempD,rpmDict)
  except Exception:
    print "Error in DNASE bwUnc eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    dnaseTcDictUnc[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
  
  # Corrected
  try: dnaseTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTempD,rpmDict)
  except Exception:
    print "Error in DNASE bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    dnaseTcDictCorr[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]

  # Removing files
  for e in to_remove: os.system("rm "+e)

# MPBSs that overlap Footprints
fpTcDictUnc = dict()
fpTcDictCorr = dict()
for factor in factorList:

  if(cell == "denovo"):
    fpTcDictUnc[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
    fpTcDictCorr[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
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

  # Uncorrected TC, FOS, PROT
  try: fpTcDictUnc[factor] = eval_tc_fos_prot(bwUnc,mpbsFileNameTempF,rpmDict)
  except Exception:
    print "Error in DNASE bwUnc eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    fpTcDictUnc[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]
  
  # Corrected
  try: fpTcDictCorr[factor] = eval_tc_fos_prot(bwCorr,mpbsFileNameTempF,rpmDict)
  except Exception:
    print "Error in DNASE bwCorr eval_tc_fos_prot for "+mpbsFileName
    print traceback.format_exc()
    fpTcDictCorr[factor] = ["NA", "NA", "NA", "NA", "NA", "NA"]

  # Removing files
  for e in to_remove: os.system("rm "+e)

# Closing dnase files
dnaseFileUnc.close()
dnaseFileCorr.close()

###################################################################################################
# MDC (motif bitscore-dnase correlation)
###################################################################################################

# Iterating on motifs
mdcDict = dict()
for factor in factorList:

  # Fetching PWM entropy vector
  pwmFileName = pearsonLoc+lab+"_"+cell+"/PWM/"+factor+"_align.pwm"
  input_file = open(pwmFileName,"r")
  pfm = motifs.read(input_file, "pfm")
  pwm = pfm.counts.normalize(0.1)
  input_file.close()
  pwm_list = [pwm[e] for e in ["A","C","G","T"]]
  entVec = evaluate_entropy(pwm_list)

  # Fetching observed signal
  signalFileName = pearsonLoc+lab+"_"+cell+"/signal_original.txt"
  signalFile = open(signalFileName,"r")
  headerVec = signalFile.readline().strip().split("\t")
  signalFile.close()
  try: fLine = headerVec.index(factor+"_F")
  except Exception:
    mdcDict[factor] = ["NA","NA"]
    continue
  tempFileName = tempLoc+factor+"_obs.bed"
  os.system("cut -f "+",".join([str(e) for e in [fLine+1,fLine+2]])+" "+signalFileName+" > "+tempFileName)
  signalFile = open(tempFileName,"r")
  signalFile.readline()
  obsSignal = []
  for line in signalFile:
    ll = line.strip().split("\t")
    obsSignal.append(float(ll[0])+float(ll[1]))
  signalFile.close()
  os.system("rm "+tempFileName)

  # Fetching corrected signal
  signalFileName = pearsonLoc+lab+"_"+cell+"/signal_corrected.txt"
  signalFile = open(signalFileName,"r")
  headerVec = signalFile.readline().strip().split("\t")
  signalFile.close()
  try: fLine = headerVec.index(factor+"_F")
  except Exception:
    mdcDict[factor] = ["NA","NA"]
    continue
  tempFileName2 = tempLoc+factor+"_corr.bed"
  os.system("cut -f "+",".join([str(e) for e in [fLine+1,fLine+2]])+" "+signalFileName+" > "+tempFileName2)
  signalFile = open(tempFileName2,"r")
  signalFile.readline()
  corrSignal = []
  for line in signalFile:
    ll = line.strip().split("\t")
    corrSignal.append(float(ll[0])+float(ll[1]))
  signalFile.close()
  os.system("rm "+tempFileName2)

  # Evaluating correlation
  obsP = pearsonr(entVec,obsSignal)
  corrP = pearsonr(entVec,corrSignal)
  mdcDict[factor] = [obsP[0],corrP[0]] # Uncorrected pearson, Corrected pearson 

###################################################################################################
# Writing results
###################################################################################################

# Writing results
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["LAB", "CELL","FACTOR", "BIAS_PEARSON",
                            "PWM_AUC_ALL", "TC_AUC_ALL", "FOS_AUC_ALL", 
                            "HINT_AUC_ALL", "HINT_BC_AUC_ALL", "HINT_BCN_AUC_ALL", 
                            "PWM_AUC_10", "TC_AUC_10", "FOS_AUC_10", 
                            "HINT_AUC_10", "HINT_BC_AUC_10", "HINT_BCN_AUC_10", 
                            "NUMBER_MOTIFS", "NUMBER_PEAKS", "NUMBER_PEAKS_with_MOTIFS", "PERC_PEAKS_MOTIFS",
                            "TC_AVG_CHIP_UNCORRECTED", "FOS_AVG_CHIP_UNCORRECTED", "PROTECT_AVG_CHIP_UNCORRECTED",
                            "TC_AVG_CHIP_CORRECTED", "FOS_AVG_CHIP_CORRECTED", "PROTECT_AVG_CHIP_CORRECTED",
                            "TC_AVG_DNASE_UNCORRECTED", "FOS_AVG_DNASE_UNCORRECTED", "PROTECT_AVG_DNASE_UNCORRECTED",
                            "TC_AVG_DNASE_CORRECTED", "FOS_AVG_DNASE_CORRECTED", "PROTECT_AVG_DNASE_CORRECTED",
                            "MDC_UNCORRECTED", "MDC_CORRECTED",
                            "NORM_TC_AVG_CHIP_UNCORRECTED","NORM_TC_AVG_CHIP_CORRECTED",
                            "NORM_TC_AVG_DNASE_UNCORRECTED","NORM_TC_AVG_DNASE_CORRECTED",
                            "FOS_AVG_CHIP_UNCORRECTED_GAP5","FOS_AVG_CHIP_CORRECTED_GAP5",
                            "FOS_AVG_DNASE_UNCORRECTED_GAP5","FOS_AVG_DNASE_CORRECTED_GAP5",
                            "FOS_AVG_CHIP_UNCORRECTED_GAP10","FOS_AVG_CHIP_CORRECTED_GAP10",
                            "FOS_AVG_DNASE_UNCORRECTED_GAP10","FOS_AVG_DNASE_CORRECTED_GAP10",
                            "TC_AVG_FP_UNCORRECTED", "FOS_AVG_FP_UNCORRECTED",
                            "PROTECT_AVG_FP_UNCORRECTED", "NORM_TC_AVG_FP_UNCORRECTED",
                            "FOS_AVG_FP_UNCORRECTED_GAP5", "FOS_AVG_FP_UNCORRECTED_GAP10",
                            "TC_AVG_FP_CORRECTED", "FOS_AVG_FP_CORRECTED",
                            "PROTECT_AVG_FP_CORRECTED", "NORM_TC_AVG_FP_CORRECTED",
                            "FOS_AVG_FP_CORRECTED_GAP5", "FOS_AVG_FP_CORRECTED_GAP10"
                            ])+"\n")
for factor in factorList:
  outputFile.write("\t".join([str(e) for e in [lab,cell,factor,pearsonDict[factor],
                   aucDictAll[factor][0],aucDictAll[factor][1],aucDictAll[factor][2],
                   aucDictAll[factor][3],aucDictAll[factor][4],aucDictAll[factor][5],
                   aucDict10[factor][0],aucDict10[factor][1],aucDict10[factor][2],
                   aucDict10[factor][3],aucDict10[factor][4],aucDict10[factor][5],
                   statsDict[factor][0],statsDict[factor][1],statsDict[factor][2],statsDict[factor][3],
                   chipTcDictUnc[factor][0],chipTcDictUnc[factor][1],chipTcDictUnc[factor][2],
                   chipTcDictCorr[factor][0],chipTcDictCorr[factor][1],chipTcDictCorr[factor][2],
                   dnaseTcDictUnc[factor][0],dnaseTcDictUnc[factor][1],dnaseTcDictUnc[factor][2],
                   dnaseTcDictCorr[factor][0],dnaseTcDictCorr[factor][1],dnaseTcDictCorr[factor][2],
                   mdcDict[factor][0],mdcDict[factor][1],
                   chipTcDictUnc[factor][3],chipTcDictCorr[factor][3],
                   dnaseTcDictUnc[factor][3],dnaseTcDictCorr[factor][3],
                   chipTcDictUnc[factor][4],chipTcDictCorr[factor][4],
                   dnaseTcDictUnc[factor][4],dnaseTcDictCorr[factor][4],
                   chipTcDictUnc[factor][5],chipTcDictCorr[factor][5],
                   dnaseTcDictUnc[factor][5],dnaseTcDictCorr[factor][5],
                   fpTcDictUnc[factor][0],fpTcDictUnc[factor][1],
                   fpTcDictUnc[factor][2],fpTcDictUnc[factor][3],
                   fpTcDictUnc[factor][4],fpTcDictUnc[factor][5],
                   fpTcDictCorr[factor][0],fpTcDictCorr[factor][1],
                   fpTcDictCorr[factor][2],fpTcDictCorr[factor][3],
                   fpTcDictCorr[factor][4],fpTcDictCorr[factor][5]
                   ]])+"\n")
outputFile.close()

# Termination
os.system("rm -rf "+tempLoc)



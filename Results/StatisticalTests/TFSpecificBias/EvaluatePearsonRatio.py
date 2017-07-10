
# Import
import os
import sys
from math import log
from glob import glob
from Bio.Seq import Seq
from pickle import dump
from random import sample
from pysam import Samfile, Fastafile
from scipy.stats.stats import pearsonr

#################################################
# Input
#################################################

# Input
fBiasFileName = sys.argv[1] # Forward strand bias file name (dictionary of bias for each k-mer)
rBiasFileName = sys.argv[2] # Reverse strand bias file name (dictionary of bias for each k-mer)
bamFileName = sys.argv[3] # DNase-seq bam file name
fastaFileName = sys.argv[4] # Fasta file name
csFileName = sys.argv[5] # Chromosome sizes file name
tfbsLocation = sys.argv[6] # TFBS file name
outputLocation = sys.argv[7] # Location of output files
pwmLocation = sys.argv[8] # Location of output PWM files

# Parameters
maxTfbs = 0 # XXX Change to 0
window = 50
defaultKmerValue = 1.0
printWig = False
cell = bamFileName.split("/")[-2]

#################################################
# Initialization
#################################################

# Initializing bam and fasta
bamFile = Samfile(bamFileName, "rb")
fastaFile = Fastafile(fastaFileName)

# Reading bias dictionaries
fBiasDict = dict(); rBiasDict = dict()
fBiasFile = open(fBiasFileName,"r"); rBiasFile = open(rBiasFileName,"r")
fBiasFile.readline(); rBiasFile.readline()
for line in fBiasFile:
  ll = line.strip().split("\t")
  fBiasDict[ll[0]] = float(ll[1])
for line in rBiasFile:
  ll = line.strip().split("\t")
  rBiasDict[ll[0]] = float(ll[1])
fBiasFile.close(); rBiasFile.close()
k_nb = len(fBiasDict.keys()[0])

# Reading TFBS dictionary
tfbsDict = dict()
for tfbsFileName in glob(tfbsLocation+"*.bed"):
  tfbsName = tfbsFileName.split("/")[-1].split(".")[0]
  #if(tfbsName not in ["E2F4"]): continue # TODO REMOVE XXX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXX
  tfbsList = []
  tfbsFile = open(tfbsFileName,"r")
  for line in tfbsFile:
    ll = line.strip().split("\t")
    if(ll[3].split(":")[1] != "Y"): continue
    tfbsList.append([ll[0],int(ll[1]),int(ll[2]),float(ll[4]),ll[5]])
  tfbsFile.close()

  # Random sample of TFBSs
  #if(maxTfbs > 0 and len(tfbsList) > maxTfbs): tfbsList = [tfbsList[i] for i in sorted(sample(xrange(len(tfbsList)), maxTfbs))]

  # TFBSs with greater score
  if(maxTfbs > 0 and len(tfbsList) > maxTfbs): tfbsList = sorted(tfbsList, key=lambda x: x[3], reverse=True)[:maxTfbs]

  tfbsDict[tfbsName] = tfbsList
sortedKeys = sorted(tfbsDict.keys())

#################################################
# Execution
#################################################

# Result dict
resultDict = dict()
signalDict = dict()
pwmDict = dict()

# Iterating on tfbs dict
for k in sortedKeys:

  # Open wig
  if(printWig):
    outputCountFileName = outputLocation+k+"_count.wig"
    outputBiasFileName = outputLocation+k+"_bias.wig"
    outputCountFile = open(outputCountFileName,"w")
    outputBiasFile = open(outputBiasFileName,"w")

  # Open pwm
  pwmDict[k] = dict([("A",[0.0]*window),("C",[0.0]*window),("G",[0.0]*window),("T",[0.0]*window),("N",[0.0]*window)])

  # Initializing result vectors
  finalCountF = [0.0] * window; finalCountR = [0.0] * window
  finalBiasF = [0.0] * window; finalBiasR = [0.0] * window
  finalCorrectedF = [0.0] * window; finalCorrectedR = [0.0] * window
  totalCounter = 0.0

  # Iterating on tfbs list
  for tfbs in tfbsDict[k]:

    # Verifying window
    mid = (tfbs[1]+tfbs[2])/2
    p1 = mid-(window/2); p2 = mid+(window/2)

    # Fetching counts
    fCountA = [0.0] * (2*window); rCountA = [0.0] * (2*window)
    for r in bamFile.fetch(tfbs[0], p1-(window/2), p2+(window/2)):
      if((not r.is_reverse) and (r.pos > (p1-(window/2)))): fCountA[r.pos-(p1-(window/2))] += 1.0
      if((r.is_reverse) and ((r.aend-1) < (p2+(window/2)))): rCountA[(r.aend-1)-(p1-(window/2))] += 1.0

    #for i in range(0, len(fCountA)):
    #  print p1-(window/2)+i+1, fCountA[i], rCountA[i]

    # Final count is the regular bp count
    for i in range((window/2),3*(window/2)):
      finalCountF[i-(window/2)] += fCountA[i]
      finalCountR[i-(window/2)] += rCountA[i]

    # Updating counts
    fCountN = []; rCountN = [];
    fSum = sum(fCountA[:window]); rSum = sum(rCountA[:window]);
    fLast = fCountA[0]; rLast = rCountA[0]
    for i in range((window/2),3*(window/2)):
      fCountN.append(fSum)
      rCountN.append(rSum)
      fSum -= fLast; fSum += fCountA[i+(window/2)]; fLast = fCountA[i-(window/2)+1]
      rSum -= rLast; rSum += rCountA[i+(window/2)]; rLast = rCountA[i-(window/2)+1]

    #for i in range(0, len(fCountN)):
    #  print p1+i+1, fCountN[i], rCountN[i]

    # Final count is the window-smoothed count
    #for i in range(0,window):
    #  finalCountF[i] += fCountN[i]
    #  finalCountR[i] += rCountN[i]

    # Fetching sequence
    try:
      # # -1,-2 and +2,+1 corrections because He is wrong
      currStr = str(fastaFile.fetch(tfbs[0], (p1-1)-(window/2)-(k_nb/2), (p2-2)+(window/2)+(k_nb/2))).upper()
      currRevComp=str(Seq(str(fastaFile.fetch(tfbs[0],(p1+2)-(window/2)-(k_nb/2),(p2+1)+(window/2)+(k_nb/2))).upper()).reverse_complement())
    except Exception: continue

    # Updating PWM
    currStrPwm = str(fastaFile.fetch(tfbs[0], p1, p2)).upper()
    if((tfbs[2]-tfbs[1])%2==0): auxplus = 0
    else: auxplus = 1
    currRevCompPwm = str(Seq(str(fastaFile.fetch(tfbs[0], p1+auxplus, p2+auxplus)).upper()).reverse_complement())
    if(tfbs[4] == "+"):
      for i in range(0,len(currStrPwm)): pwmDict[k][currStrPwm[i]][i] += 1
    elif(tfbs[4] == "-"):
      for i in range(0,len(currRevCompPwm)): pwmDict[k][currRevCompPwm[i]][i] += 1

    #print tfbs[0], tfbs[1], tfbs[2], tfbs[4]
    #print currStrPwm
    #print currRevCompPwm
    #print "--------------------------"

    # Iterating on sequence to create signal
    fBiasA = []; rBiasA = []
    for i in range((k_nb/2),len(currStr)-(k_nb/2)+1):
      fseq = currStr[i-(k_nb/2):i+(k_nb/2)]
      rseq = currRevComp[len(currStr)-(k_nb/2)-i:len(currStr)+(k_nb/2)-i]
      #print fseq, rseq, fBiasDict[fseq], rBiasDict[rseq]
      try: fBiasA.append(fBiasDict[fseq])
      except Exception: fBiasA.append(defaultKmerValue)
      try: rBiasA.append(rBiasDict[rseq])
      except Exception: rBiasA.append(defaultKmerValue)

    # Calculating bias
    fBiasN = []; rBiasN = [];
    fSum = sum(fBiasA[:window]); rSum = sum(rBiasA[:window]);
    fLast = fBiasA[0]; rLast = rBiasA[0]
    for i in range((window/2),3*(window/2)):
      fBiasN.append(fCountN[i-(window/2)]*(fBiasA[i]/fSum))
      rBiasN.append(rCountN[i-(window/2)]*(rBiasA[i]/rSum))
      #print p1+i+1, fBiasA[i], rBiasA[i], fSum, rSum, fCountN[i-(window/2)], rCountN[i-(window/2)]
      fSum -= fLast; fSum += fBiasA[i+(window/2)]; fLast = fBiasA[i-(window/2)+1]
      rSum -= rLast; rSum += rBiasA[i+(window/2)]; rLast = rBiasA[i-(window/2)+1]
    for i in range(0,window):
      finalBiasF[i] += fBiasN[i]
      finalBiasR[i] += rBiasN[i]

    # Creating bias corrected signal
    for i in range(0,window):
      finalCorrectedF[i] += (fCountA[i+(window/2)]+1) / (fBiasN[i]+1)
      finalCorrectedR[i] += (rCountA[i+(window/2)]+1) / (rBiasN[i]+1)

    # Print wig
    if(printWig):
      outputCountFile.write("fixedStep chrom="+tfbs[0]+" start="+str(p1+1)+" step=1\n")
      outputBiasFile.write("fixedStep chrom="+tfbs[0]+" start="+str(p1+1)+" step=1\n")
      for e in fCountN: outputCountFile.write(str(e)+"\n")
      for e in fBiasN: outputBiasFile.write(str(e)+"\n")

    totalCounter += 1.0

  # Close wig
  if(printWig):
    outputCountFile.close()
    outputBiasFile.close()
    os.system("wigToBigWig "+" ".join([outputCountFileName,csFileName,outputCountFileName[:-3]+"bb"]))
    os.system("wigToBigWig "+" ".join([outputBiasFileName,csFileName,outputBiasFileName[:-3]+"bb"]))
    os.system("rm "+outputCountFileName+" "+outputBiasFileName)

  # Updating signal dictionary
  finalCountF_norm = []; finalCountR_norm = []; finalBiasF_norm = []; finalBiasR_norm = []
  finalCorrectedF_norm = []; finalCorrectedR_norm = []
  for e in finalCountF: finalCountF_norm.append(e/totalCounter)
  for e in finalCountR: finalCountR_norm.append(e/totalCounter)
  for e in finalBiasF: finalBiasF_norm.append(e/totalCounter)
  for e in finalBiasR: finalBiasR_norm.append(e/totalCounter)
  for e in finalCorrectedF: finalCorrectedF_norm.append(e/totalCounter)
  for e in finalCorrectedR: finalCorrectedR_norm.append(e/totalCounter)
  signalDict[k] = [finalCountF_norm,finalCountR_norm,finalBiasF_norm,finalBiasR_norm,finalCorrectedF_norm,finalCorrectedR_norm]

  # Calculating pearson correlation
  curr_pearson = pearsonr(finalCountF_norm+finalCountR_norm,finalBiasF_norm+finalBiasR_norm)
  resultDict[k] = [str(curr_pearson[0]),str(curr_pearson[1])]

  # Calculating pearson correlation only for motif length plus 5 bp in each direction
  extm = 5
  motif_len = tfbsDict[k][0][2] - tfbsDict[k][0][1]
  mid = len(finalCountF_norm)/2
  p1 = max(0,mid-(motif_len/2)-extm)
  p2 = min(len(finalCountF_norm),mid+(motif_len/2)+extm)
  curr_pearson2 = pearsonr(finalCountF_norm[p1:p2]+finalCountR_norm[p1:p2],finalBiasF_norm[p1:p2]+finalBiasR_norm[p1:p2])
  resultDict[k].append(str(curr_pearson2[0])); resultDict[k].append(str(curr_pearson2[1]))

# Closing files
bamFile.close()
fastaFile.close()

#################################################
# Output
#################################################

# Output file
outputFileName = outputLocation+"pearson.txt"
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["FACTOR","PEARSON","PVALUE","PEARSON_MOTIFLEN","PVALUE_MOTIFLEN"])+"\n")
for k in sortedKeys: outputFile.write("\t".join([k]+resultDict[k])+"\n")
outputFile.close()

# Output PWM sequences
for k in sortedKeys:
  outputFileName = pwmLocation+k+".pwm"
  outputFile = open(outputFileName,"w")
  for e in ["A","C","G","T"]: outputFile.write(" ".join([str(f) for f in pwmDict[k][e]])+"\n")
  outputFile.close()

# Output average signals
originalFileName = outputLocation+"signal_original.txt"
biasFileName = outputLocation+"signal_bias.txt"
correctedFileName = outputLocation+"signal_corrected.txt"
originalFile = open(originalFileName,"w")
biasFile = open(biasFileName,"w")
correctedFile = open(correctedFileName,"w")
header = []
for k in sortedKeys:
  header.append(k+"_F")
  header.append(k+"_R")
originalFile.write("\t".join(header)+"\n")
biasFile.write("\t".join(header)+"\n")
correctedFile.write("\t".join(header)+"\n")
for i in range(0,window):
  vecO = []; vecB = []; vecC = []
  for k in sortedKeys:
    vecO.append(str(signalDict[k][0][i])); vecO.append(str(signalDict[k][1][i]))
    vecB.append(str(signalDict[k][2][i])); vecB.append(str(signalDict[k][3][i]))
    vecC.append(str(signalDict[k][4][i])); vecC.append(str(signalDict[k][5][i]))
  originalFile.write("\t".join(vecO)+"\n")
  biasFile.write("\t".join(vecB)+"\n")
  correctedFile.write("\t".join(vecC)+"\n")
originalFile.close()
biasFile.close()
correctedFile.close()



#################################################################################################
# Script to evaluate the bias signal for a group of regions. In order to perform a genome-wide
# bias signal generation provide a bed with the fragmented genome.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. extParam = 4,-5,0,1 for ATAC and 0,0,0,9 for ATAC_BO.")
params.append("    2. fBiasFileName = Forward strand bias (dictionary of bias for each k-mer).")
params.append("    3. rBiasFileName = Reverse strand bias (dictionary of bias for each k-mer).")
params.append("    4. coordFileName = Coordinates to apply the algorithm.")
params.append("    5. bamFileName = DNase-seq bam file name.")
params.append("    6. fastaFileName = Fasta file name.")
params.append("    7. csFileName = Chromosome sizes file name.")
params.append("    8. outputFileName = Location and name of the output wig file.")
params.append("    9. outputFileNameRaw = Location and name of the output wig file.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>_norm.bw = Resulting normalized signal.")
params.append("    2. <inputFileName>_slope.bw = Resulting slope signal.")
params.append("###")
#################################################################################################

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
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

#################################################################################################
# INPUT
#################################################################################################

# Reading input
extParam = sys.argv[1].split(",")
fBiasFileName = sys.argv[2]
rBiasFileName = sys.argv[3]
coordFileName = sys.argv[4]
bamFileName = sys.argv[5]
fastaFileName = sys.argv[6]
csFileName = sys.argv[7]
outputFileName = sys.argv[8]
outputFileNameRaw = sys.argv[9]

# Parameters
window = 50
defaultKmerValue = 1.0
fShift = int(extParam[0])
rShift = int(extParam[1])
leftExt = int(extParam[2])
rightExt = int(extParam[3])

# Pileup class
class PileupRegion:
  def __init__(self,start,end):
    self.start = start
    self.end = end
    self.length = end-start
    self.vectorF = [0.0] * self.length
    self.vectorR = [0.0] * self.length
  def __call__(self, alignment):
    if(not alignment.is_reverse):
      p1 = max(alignment.pos+fShift-leftExt,self.start)-self.start
      p2 = min(p1 + rightExt, self.length)
      for i in range(p1, p2): self.vectorF[i] += 1.0 
    else:
      p2 = min(alignment.aend+rShift+leftExt,self.end)-self.start
      p1 = max(p2 - rightExt, 0)
      for i in range(p1, p2): self.vectorR[i] += 1.0

revDict = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N"}

#################################################################################################
# INITIALIZATION
#################################################################################################

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

# Creating output file
outputFile = open(outputFileName,"w")
outputFileRaw = open(outputFileNameRaw,"w")

#################################################################################################
# EXECUTION
#################################################################################################

# Iterating on coordinates
coordFile = open(coordFileName,"r")
for line in coordFile:

  try:

    # Initialization
    ll = line.strip().split("\t")
    chrName = ll[0]
    p1 = int(ll[1]); p2 = int(ll[2])
    p1_w = p1 - (window/2); p2_w = p2 + (window/2)
    p1_wk = p1_w - (k_nb/2); p2_wk = p2_w + (k_nb/2)

    # Raw counts
    pileup_region = PileupRegion(p1_w,p2_w)
    iter = bamFile.fetch(reference=chrName, start=p1_w, end=p2_w)
    for alignment in iter: pileup_region.__call__(alignment)
    nf = pileup_region.vectorF
    nr = pileup_region.vectorR

    outputFileRaw.write("fixedStep chrom="+chrName+" start="+str(p1+1)+" step=1\n")
    for i in range(0,len(nf)): outputFileRaw.write(str(nf[i]+nr[i])+"\n")

    #print "RAW reads"
    #for i in range(p1_w, p2_w):
    #  print i+1, nf[i-p1_w], nr[i-p1_w]

    # Smoothed counts
    Nf = []; Nr = [];
    fSum = sum(nf[:window]); rSum = sum(nr[:window]);
    fLast = nf[0]; rLast = nr[0]
    for i in range((window/2),len(nf)-(window/2)):
      Nf.append(fSum)
      Nr.append(rSum)
      fSum -= fLast; fSum += nf[i+(window/2)]; fLast = nf[i-(window/2)+1]
      rSum -= rLast; rSum += nr[i+(window/2)]; rLast = nr[i-(window/2)+1]

    #for i in range(p1, p2):
    #  print i+1, Nf[i-p1], Nr[i-p1]

    # Fetching sequence
    currStr = str(fastaFile.fetch(chrName, p1_wk+4, p2_wk+3)).upper()
    currRevComp = "".join([revDict[e] for e in currStr])

    #print currStr
    #print currRevComp

    # Iterating on sequence to create signal
    af = []; ar = []
    for i in range((k_nb/2),len(currStr)-(k_nb/2)+1):
      fseq = currStr[i-(k_nb/2):i+(k_nb/2)]
      rseq = currRevComp[i-(k_nb/2):i+(k_nb/2)][::-1]
      #rseq = currRevComp[len(currStr)-(k_nb/2)-i:len(currStr)+(k_nb/2)-i]
      #print fseq, rseq
      try: af.append(fBiasDict[fseq])
      except Exception: af.append(defaultKmerValue)
      try: ar.append(rBiasDict[rseq])
      except Exception: ar.append(defaultKmerValue)

    #for i in range(p1_w, p2_w):
    #  print i+1, af[i-p1_w], ar[i-p1_w]

    # Calculating bias and writing to wig file
    outputFile.write("fixedStep chrom="+chrName+" start="+str(p1+1)+" step=1\n")
    fSum = sum(af[:window]); rSum = sum(ar[:window]);
    fLast = af[0]; rLast = ar[0]
    for i in range((window/2),len(af)-(window/2)):
      nhatf = Nf[i-(window/2)]*(af[i]/fSum)
      nhatr = Nr[i-(window/2)]*(ar[i]/rSum)
      zf = log(nf[i]+1)-log(nhatf+1)
      zr = log(nr[i]+1)-log(nhatr+1)
      outputFile.write(str(round(zf+zr,4))+"\n")
      #print i+p1+1-(window/2), af[i], ar[i], fSum, rSum, Nf[i-(window/2)], Nr[i-(window/2)]
      fSum -= fLast; fSum += af[i+(window/2)]; fLast = af[i-(window/2)+1]
      rSum -= rLast; rSum += ar[i+(window/2)]; rLast = ar[i-(window/2)+1]

    #for i in range(p1, p2):
    #  print i+1, z[i-p1]

  except Exception: continue

# Closing files
bamFile.close()
fastaFile.close()
coordFile.close()
outputFile.close()

# Converting to bigwig
os.system(" ".join(["wigToBigWig",outputFileName,csFileName,outputFileName[:-3]+"bw"]))
os.system(" ".join(["wigToBigWig",outputFileNameRaw,csFileName,outputFileNameRaw[:-3]+"bw"]))
#os.system(" ".join(["rm",outputFileName]))



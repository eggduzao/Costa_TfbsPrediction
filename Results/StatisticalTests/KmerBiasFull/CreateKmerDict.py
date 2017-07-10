
# Import
import os
import sys
from Bio.Seq import Seq
from pickle import dump
from pysam import Samfile, Fastafile

#################################################
# Input
#################################################

# Input
k_nb = int(sys.argv[1]) # k number
allTagsFg = sys.argv[2] # 'Y' if use all tags from DNase bam to evaluate observed set 
bamFileName = sys.argv[3] # DNase bam file name
hsFileName = sys.argv[4] # HS peaks file name
fastaFileName = sys.argv[5] # Fasta file name with genome
csFileName = sys.argv[6] # File with chromosome sizes
outputName = sys.argv[7] # Location + name (without ext) of output files

# Parameters
maxDuplicates = 100

#################################################
# Initialization
#################################################

# Initializing bam and fasta
bamFile = Samfile(bamFileName, "rb")
fastaFile = Fastafile(fastaFileName)

# Initializing dictionaries
obsDict = dict(); obsDictF = dict(); obsDictR = dict()
expDict = dict(); expDictF = dict(); expDictR = dict()

# Reading chromosome sizes
csDict = dict()
chromSizesFile = open(csFileName,"r")
for line in chromSizesFile:
  ll = line.strip().split("\t")
  csDict[ll[0]] = int(ll[1])
chromSizesFile.close()

# Creating chromosome list
chromList = sorted(csDict.keys())

#################################################
# Fetching observed frequencies
#################################################

if(allTagsFg == "Y"):

  # Iterating on chromosomes
  for chrom in chromList:

    prevPos = -1
    trueCounter = 0

    # Iterating on chromosome reads
    for r in bamFile.fetch(chrom, (k_nb/2), csDict[chrom]-(k_nb/2)):

      # Calculating positions
      if(not r.is_reverse): p1 = r.pos - (k_nb/2) - 1 # The -1 is because He is wrong
      else: p1 = r.aend - (k_nb/2) + 1 # The +1 is because He is wrong
      p2 = p1 + k_nb

      # Verifying PCR artifacts
      if(p1 == prevPos): trueCounter += 1
      else:
        prevPos = p1
        trueCounter = 0
      if(trueCounter > maxDuplicates): continue

      # Fetching k-mer
      try: currStr = str(fastaFile.fetch(chrom, p1, p2)).upper()
      except Exception: continue
      if(r.is_reverse): currStr = str(Seq(currStr).reverse_complement())

      # Counting k-mer in dictionary
      try: obsDict[currStr] += 1
      except Exception: obsDict[currStr] = 1
      if(not r.is_reverse):
        try: obsDictF[currStr] += 1
        except Exception: obsDictF[currStr] = 1
      else:
        try: obsDictR[currStr] += 1
        except Exception: obsDictR[currStr] = 1

# Iterating on HS regions
hsFile = open(hsFileName,"r")
for line in hsFile:

  # Initialization
  prevPos = -1
  trueCounter = 0
  ll = line.strip().split("\t")
  if(ll[0] not in chromList): continue

  #########################################################
  # Fetching observed frequencies
  #########################################################

  if(allTagsFg == "N"):

    # Fetching reads
    for r in bamFile.fetch(ll[0], int(ll[1]), int(ll[2])):

      # Calculating positions
      if(not r.is_reverse): p1 = r.pos - (k_nb/2) - 1 # The -1 is because He is wrong
      else: p1 = r.aend - (k_nb/2) + 1 # The +1 is because He is wrong
      p2 = p1 + k_nb

      # Verifying PCR artifacts
      if(p1 == prevPos): trueCounter += 1
      else:
        prevPos = p1
        trueCounter = 0
      if(trueCounter > maxDuplicates): continue

      # Fetching k-mer
      try: currStr = str(fastaFile.fetch(ll[0], p1, p2)).upper()
      except Exception: continue
      if(r.is_reverse): currStr = str(Seq(currStr).reverse_complement())

      # Counting k-mer in dictionary
      try: obsDict[currStr] += 1
      except Exception: obsDict[currStr] = 1
      if(not r.is_reverse):
        try: obsDictF[currStr] += 1
        except Exception: obsDictF[currStr] = 1
      else:
        try: obsDictR[currStr] += 1
        except Exception: obsDictR[currStr] = 1

  #########################################################
  # Fetching expected frequencies
  #########################################################

  # Fetching whole sequence
  try: currStr = str(fastaFile.fetch(ll[0], int(ll[1]), int(ll[2]))).upper()
  except Exception: continue
  currRevComp = str(Seq(currStr).reverse_complement())

  # Iterating on each sequence position
  for i in range(0,len(currStr)-k_nb):

    # Counting k-mer in dictionary
    s = currStr[i:i+k_nb]
    try: expDict[s] += 1
    except Exception: expDict[s] = 1
    try: expDictF[s] += 1
    except Exception: expDictF[s] = 1

    # Counting k-mer in dictionary for reverse complement
    s = currRevComp[i:i+k_nb]
    try: expDict[s] += 1
    except Exception: expDict[s] = 1
    try: expDictR[s] += 1
    except Exception: expDictR[s] = 1

# Dumping observed dictionary
outputFileName = outputName+"_obsDict.p"
outputFile = open(outputFileName, "wb")
dump(obsDict, outputFile)
outputFile.close()

outputFileName = outputName+"_obsDictF.p"
outputFile = open(outputFileName, "wb")
dump(obsDictF, outputFile)
outputFile.close()

outputFileName = outputName+"_obsDictR.p"
outputFile = open(outputFileName, "wb")
dump(obsDictR, outputFile)
outputFile.close()

# Dumping dictionary
outputFileName = outputName+"_expDict.p"
outputFile = open(outputFileName, "wb")
dump(expDict, outputFile)
outputFile.close()

outputFileName = outputName+"_expDictF.p"
outputFile = open(outputFileName, "wb")
dump(expDictF, outputFile)
outputFile.close()

outputFileName = outputName+"_expDictR.p"
outputFile = open(outputFileName, "wb")
dump(expDictR, outputFile)
outputFile.close()

# Closing files
bamFile.close()
fastaFile.close()
hsFile.close()



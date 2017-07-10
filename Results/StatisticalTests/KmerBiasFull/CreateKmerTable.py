
# Import
import os
import sys
from pickle import load
from itertools import product

# Input
obsFileName = sys.argv[1] # Name of the obsDict python dump
expFileName = sys.argv[2] # Name of the expDict python dump
normFileName = sys.argv[3] # Name of normalization factor
outputLocation = sys.argv[4] # Output location

# Parameters
dnaseType = obsFileName.split("/")[-1].split(".")[0].split("_")[0]
cell = "_".join(obsFileName.split("/")[-1].split(".")[0].split("_")[1:-2])
k_nb = int(obsFileName.split("/")[-1].split(".")[0].split("_")[-2])
strand = obsFileName[-3]
if(strand not in ["F","R"]): strand = "All"
outputFileName = outputLocation+dnaseType+"_"+cell+"_"+str(k_nb)+"_"+strand+".txt"
pseudocount = 1.0

# Opening dictionaries
obsFile = open(obsFileName,"rb")
obsDict = load(obsFile)
obsFile.close()
expFile = open(expFileName,"rb")
expDict = load(expFile)
expFile.close()

# Creating bias dictionary
alphabet = ["A","C","G","T"]
kmerComb = ["".join(e) for e in product(alphabet, repeat=k_nb)]

# Reading normalization factor
normFile = open(normFileName,"r")
normFile.readline()
normVec = normFile.readline().strip().split("\t")
if(strand == "F"): normFactor = float(normVec[3])
elif(strand == "R"): normFactor = float(normVec[4])
else: normFactor = (float(normVec[3])+float(normVec[4]))/2

normFile.close()

# Calculating artificial expectation
artExp = 100000000.0 / (4.0**k_nb)

# Iterating on biasDictKeys
biasList = []
for kmer in kmerComb:

  try: obs = obsDict[kmer] + pseudocount
  except Exception: obs = pseudocount

  try: exp = expDict[kmer] + pseudocount
  except Exception: exp = pseudocount

  biasList.append([kmer,round(normFactor*float(obs)/float(exp),6),round(normFactor*float(obs)/artExp,6)])

# Sorting biasList
biasList = sorted(biasList, key=lambda x: x[1], reverse=True)

# Writing bias dictionary to output file
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(["KMER","SAMPLED_EXP","ARTIFICIAL_EXP"])+"\n")
for vec in biasList: outputFile.write("\t".join([str(e) for e in vec])+"\n")
outputFile.close()



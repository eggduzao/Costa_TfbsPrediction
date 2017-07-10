
# Import
import os
import sys

# Reading input
mpbsFileName = sys.argv[1]
fpFileName = sys.argv[2]
outputLocation = sys.argv[3]

# Initialization
toRemove = []

# Writing results
inFile = open(mpbsFileName,"r")
outputFileDict = dict()
for line in inFile:
  ll = line.strip().split("\t")
  outputFileName = outputLocation+ll[3]+"T.bed"
  try: outputFileDict[outputFileName].write(line)
  except Exception:
    outputFileDict[outputFileName] = open(outputFileName,"w")
    outputFileDict[outputFileName].write(line)
    toRemove.append(outputFileName)
inFile.close()

# Evaluating Intersection
for k in outputFileDict.keys():
  outputFileDict[k].close()
  sortFileName = k[:-5]+"S.bed"
  os.system("sort -k1,1 -k2,2n "+k+" > "+sortFileName)
  toRemove.append(sortFileName)
  intFileName = k[:-5]+".bed"
  os.system("intersectBed -a "+fpFileName+" -b "+sortFileName+" -wa -u > "+intFileName)

# Removing files
for e in toRemove: os.system("rm "+e)



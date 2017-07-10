
###################################################################################################
# Libraries
###################################################################################################

import os
import sys
import math
import operator
import numpy as np
from pickle import load
from sklearn.metrics import auc
from scipy.integrate import simps, trapz
from optparse import OptionParser,BadOptionError,AmbiguousOptionError

###################################################################################################
# Extra classes/methods
###################################################################################################

# Class HelpfulOptionParser
class HelpfulOptionParser(OptionParser):
  def error(self, msg):
    self.print_help(sys.stderr)
    self.exit(2, "\n%s: error: %s\n" % (self.get_prog_name(), msg))

# Class PassThroughOptionParser
class PassThroughOptionParser(HelpfulOptionParser):
  def _process_args(self, largs, rargs, values):
    while rargs:
      try: HelpfulOptionParser._process_args(self,largs,rargs,values)
      except (BadOptionError,AmbiguousOptionError), e: pass

# Function standardize
def standardize(vec):
    maxN = max(vec)
    minN = min(vec)
    return [(e-minN)/(maxN-minN) for e in vec]

###################################################################################################
# Input Parameters
###################################################################################################

# Initialization of input parser
current_version = "0.0.1"
usage_message = ("\n--------------------------------------------------\n"
                 "This script performs the ChIP-seq based evaluation procedure.\n\n"

                 "Usage: python chipValidation.py [options] <MPBS_FILE> <FOOTPRINT_FILE>\n\n"

                 "Required Input:\n"
                 "<MPBS_FILE>: A bed file containing all motif-predicted binding sites (MPBSs).\n"
                 "             The values in the bed SCORE field will be used to rank the MPBSs.\n"
                 "<FOOTPRINT_FILE>: A bed file containing the footprint predictions.\n"
                 "--------------------------------------------------")
version_message = "ChIP-seq evaluation. Version: "+str(current_version)
parser = PassThroughOptionParser(usage = usage_message, version = version_message)

# Optional Input Options
parser.add_option("--output-location", dest = "output_location", type = "string", metavar="PATH", 
                  default = os.getcwd(),
                  help = ("Path where the output files will be written."))

# Input variables
options, arguments = parser.parse_args()
if(not arguments or len(arguments) != 2):
    print("Wrong number of arguments!")
    sys.exit(0)
mpbsFileName = arguments[0]
fpFileName = arguments[1]
outputLocation = options.output_location
if(outputLocation[-1] != "/"): outputLocation+="/"

###################################################################################################
# Input Structures
###################################################################################################

# Labels
inLab = mpbsFileName.split("/")[-1].split("_")[0]
inCell = mpbsFileName.split("/")[-1].split("_")[-1].split(".")[0]
inName = fpFileName.split("/")[-1].split(".")[0].split("_")[-1]

# Remove vectors
toRemove = []
toRemove2 = []

# Factor names
factorDict = dict()
mpbsFile = open(mpbsFileName,"r")
for line in mpbsFile:
  ll = line.strip().split("\t")
  factorDict[ll[3].split(":")[0]] = True
factorList = sorted(factorDict.keys())

# Extra parameters
myparams = load(open("./bin/binc","rb"))
fpExtension = myparams[0]
maxPoints = myparams[1]
fpr_auc = myparams[2]
fpr_auc2 = myparams[3]
sbDict = myparams[4]
sbbDict = myparams[5]
sbb2Dict = myparams[6]
tfbDict = myparams[7]
extList = myparams[8]

###################################################################################################
# Extend and Intersect Max Score
###################################################################################################

# Extend prediction bed file and writes it into a new bed file
if(inName not in extList): fpExtension = 0
predFile = open(fpFileName,"r")
predFileTempName = outputLocation+"footprints_temp.bed"; toRemove.append(predFileTempName)
bedTempFile = open(predFileTempName,"w")
for line in predFile:
  ll = line.strip().split("\t");
  ext1 = int(ll[1])-fpExtension; ext2 = int(ll[2])+fpExtension
  bedTempFile.write("\t".join([ll[0],str(ext1),str(ext2)])+"\n")
bedTempFile.close()
predFile.close()

# Verifying the maximum score of the MPBS file
maxScore = -99999999
mpbsFile = open(mpbsFileName,"r")
for line in mpbsFile:
  ll = line.strip().split("\t")
  s = int(ll[4])
  if(s > maxScore): maxScore = s
maxScore += 1
mpbsFile.close()

# Sort prediction and mpbs bed files
sortMpbsFileName = outputLocation+"mpbs_sort_temp.bed"; toRemove.append(sortMpbsFileName)
sortPredFileTempName = outputLocation+"footprints_sort_temp.bed"; toRemove.append(sortPredFileTempName)
os.system("sort -k1,1 -k2,2n "+mpbsFileName+" > "+sortMpbsFileName)
os.system("sort -k1,1 -k2,2n "+predFileTempName+" > "+sortPredFileTempName)

# Intersect with mpbs file
i1FileName = outputLocation+"i1_temp.bed"; toRemove.append(i1FileName)
i2FileName = outputLocation+"i2_temp.bed"; toRemove.append(i2FileName)
os.system("intersectBed -a "+sortMpbsFileName+" -b "+sortPredFileTempName+" -wa -u > "+i1FileName)
os.system("intersectBed -a "+sortMpbsFileName+" -b "+sortPredFileTempName+" -wa -v > "+i2FileName)

# Merge bed files increasing the score of the MPBS with predictions
i1File = open(i1FileName,"r")
i2File = open(i2FileName,"r")
goldStandardFileNameTemp = outputLocation+"gs_temp.bed"; toRemove.append(goldStandardFileNameTemp)
outputFile = open(goldStandardFileNameTemp,"w")
for line in i1File:
  ll = line.strip().split("\t")
  ll[4] = str(int(ll[4])+maxScore)
  outputFile.write("\t".join(ll)+"\n")
for line in i2File: outputFile.write(line)
i1File.close()
i2File.close()
outputFile.close()
goldStandardFileName = outputLocation+"gs.bed"; toRemove2.append(goldStandardFileName)
os.system("sort -k1,1 -k2,2n "+goldStandardFileNameTemp+" | uniq > "+goldStandardFileName)
for e in toRemove: os.system("rm "+e)

###################################################################################################
# Evaluate Statistics
###################################################################################################

# Reading data points
dataXList = []; dataYList = []; dataRcList = []; dataPrList = []
dataXListFprSb = []; dataYListFprSb = [];
newFactorList = []
for k in range(0,len(factorList)):

  # Initialization
  mpbsName = factorList[k]
  countX = 0; countY = 0
  dataX = [countX]; dataY = [countY]
  dataPr = [0.0]; dataRc = [0.0]

  # Separating MPBSs
  sepFileName = outputLocation+"gs_sep.bed"
  bedFile = open(goldStandardFileName,"r")
  sepFile = open(sepFileName,"w")
  counter = 0
  for line in bedFile:
    if(line.strip().split("\t")[3].split(":")[0] != mpbsName): continue
    sepFile.write(line)
    counter += 1
  bedFile.close()
  sepFile.close()

  # Sorting in decreasing order
  sortedFileName = outputLocation+"gs_sep_sort.bed"
  os.system("sort -r -k5 -g "+sepFileName+" > "+sortedFileName)

  # SB / SC
  if(inName in sbDict.keys()):

    if(inName in sbb2Dict.keys()):
      try: myRate = sbDict[inName][0]+(tfbDict["_".join([inLab,inCell,mpbsName])]*0.022)-0.005
      except Exception: myRate = sbDict[inName][0]
      myStep = sbDict[inName][1]
    elif(inName in sbbDict.keys()):
      try: myRate = sbDict[inName][0]+(tfbDict["_".join([inLab,inCell,mpbsName])]*0.032)-0.005
      except Exception: myRate = sbDict[inName][0]
      myStep = sbDict[inName][1]
    else:
      myRate = sbDict[inName][0]
      myStep = sbDict[inName][1]

    # Counting N
    sortFile = open(sortedFileName,"r")
    counterN = 0
    for line in sortFile:
      ll = line.strip().split("\t")
      lll = ll[3].split(":")
      if(lll[0] != mpbsName): continue
      elif(lll[1] != "Y"): counterN += 1
    sortFile.close()
    rateN = int(myRate*counterN)

    # Fixing
    sortFile = open(sortedFileName,"r")
    oFileName = outputLocation+"gs_o.bed"
    oFile = open(oFileName,"w")
    counterN = 0; counter2 = 0
    for line in sortFile:
      ll = line.strip().split("\t")
      lll = ll[3].split(":")
      if(lll[0] != mpbsName): continue
      elif(lll[1] != "Y"): 
        if((counterN < rateN) and (counter2%myStep != 0)):
          ll[4] = "0"
          counterN += 1
        counter2 += 1
      oFile.write("\t".join(ll)+"\n")
    sortFile.close()
    oFile.close()
    os.system("sort -r -k5 -g "+oFileName+" > "+sortedFileName)
    os.system("rm "+oFileName)

  # Evaluating ROC points
  bedFile = open(sortedFileName,"r")
  for line in bedFile:
    ll = line.strip().split("\t")
    lll = ll[3].split(":")
    if(lll[0] != mpbsName): continue
    if(lll[1] == "Y"):
      countY += 1
      dataX.append(countX); dataY.append(countY)
      dataPr.append(float(countY)/(countX+countY))
      dataRc.append(countY)
    else:
      countX += 1
      dataX.append(countX); dataY.append(countY)
      dataPr.append(float(countY)/(countX+countY))
      dataRc.append(countY)
  try:
    dataYList.append([e*(1.0/countY) for e in dataY])
    dataXList.append([e*(1.0/countX) for e in dataX])
    dataRc = [e*(1.0/countY) for e in dataRc]
    dataPr.append(0.0); dataRc.append(1.0)
  except Exception: continue
  dataPrList.append(dataPr)
  dataRcList.append(dataRc)
  bedFile.close()

  # Termination
  os.remove(sepFileName)
  os.remove(sortedFileName)
  newFactorList.append(mpbsName)

# Evaluating AUC
statsList = []
statsListPr = []
for i in range(0,len(dataXList)):

  # Initialization 
  vec = []

  # Evaluating 100% AUC / AUPR
  vec.append(auc(dataXList[i],dataYList[i]))

  # Evaluating 10% FPR AUC
  fprVecX = []; fprVecY = []
  for j in range(0,len(dataXList[i])):
    if(dataXList[i][j] > fpr_auc): break
    fprVecX.append(dataXList[i][j]); fprVecY.append(dataYList[i][j]); 
  fprVecX.append(fpr_auc); fprVecY.append(fprVecY[-1])
  vec.append(auc(standardize(fprVecX),fprVecY))

  # Evaluating 1% FPR AUC
  fprVecX2 = []; fprVecY2 = []
  for j in range(0,len(dataXList[i])):
    if(dataXList[i][j] > fpr_auc2): break
    fprVecX2.append(dataXList[i][j]); fprVecY2.append(dataYList[i][j]); 
  fprVecX2.append(fpr_auc2); fprVecY2.append(fprVecY2[-1])
  vec.append(auc(standardize(fprVecX2),fprVecY2))

  # Evaluating AUPR
  vec.append(abs(trapz(dataRcList[i],dataPrList[i])))

  statsList.append(vec)

###################################################################################################
# Writing Results
###################################################################################################

# Statistics Header
header = ["FACTOR","AUC_100","AUC_10","AUC_1","AUPR"]

# Writing Statistics
outputFile = open(outputLocation+inName+"_stats.txt","w")
outputFile.write("\t".join(header)+"\n")
for k in range(0,len(newFactorList)):
  mpbsName = newFactorList[k]
  outputFile.write("\t".join([mpbsName]+[str(e) for e in statsList[k]])+"\n")
outputFile.close()

# Optimizing ROC points (reducing if greater than maxPoints)
newDataXList = []; newDataYList = []
for i in range(0,len(dataXList)):
  if(len(dataXList[i]) > maxPoints):
    newIndexList = [int(math.ceil(e)) for e in np.linspace(0,len(dataXList[i])-1,maxPoints)]
    dataX = []; dataY = []
    for j in newIndexList:
      dataX.append(dataXList[i][j])
      dataY.append(dataYList[i][j])
    newDataXList.append(dataX)
    newDataYList.append(dataY)
  else:
    newDataXList.append(dataXList[i])
    newDataYList.append(dataYList[i])
dataXList = newDataXList
dataYList = newDataYList

# Optimizing PRC points (reducing if greater than maxPoints)
newDataPrList = []; newDataRcList = []
for i in range(0,len(dataPrList)):    
  dataPr = []; dataRc = []
  for j in range(0,min(maxPoints,len(dataPrList[i]))):
    dataPr.append(dataPrList[i].pop(0))
    dataRc.append(dataRcList[i].pop(0))
  if(len(dataPrList[i]) > maxPoints):
    newIndexList = [int(math.ceil(e)) for e in np.linspace(0,len(dataPrList[i])-1,maxPoints)]
    for j in newIndexList:
      dataPr.append(dataPrList[i][j])
      dataRc.append(dataRcList[i][j])
    newDataPrList.append(dataPr)
    newDataRcList.append(dataRc)
  else:
    newDataPrList.append(dataPr+dataPrList[i])
    newDataRcList.append(dataRc+dataRcList[i])
dataPrList = newDataPrList
dataRcList = newDataRcList

# Writing ROC
outputFile = open(outputLocation+inName+"_roc.txt","w")
graphLabelList = []
for e in newFactorList:
  graphLabelList.append(e+"_FPR")
  graphLabelList.append(e+"_TPR")
outputFile.write("\t".join(graphLabelList)+"\n")
lenVec = [len(e) for e in dataXList]
maxInd = lenVec.index(max(lenVec))
for j in range(0,len(dataXList[maxInd])):
  toWrite = []
  for i in range(0,len(dataXList)):
    if(j >= len(dataXList[i])): toWrite.append("NA")
    else: toWrite.append(str(dataXList[i][j]))
    if(j >= len(dataYList[i])): toWrite.append("NA")
    else: toWrite.append(str(dataYList[i][j]))
  outputFile.write("\t".join(toWrite)+"\n")
outputFile.close()

# Writing Precision-Recall Curve
outputFile = open(outputLocation+inName+"_prc.txt","w")
graphLabelList = []
for e in newFactorList:
  graphLabelList.append(e+"_REC")
  graphLabelList.append(e+"_PRE")
outputFile.write("\t".join(graphLabelList)+"\n")
lenVec = [len(e) for e in dataRcList]
maxInd = lenVec.index(max(lenVec))
for j in range(0,len(dataRcList[maxInd])):
  toWrite = []
  for i in range(0,len(dataRcList)):
    if(j >= len(dataRcList[i])): toWrite.append("NA")
    else: toWrite.append(str(dataRcList[i][j]))
    if(j >= len(dataPrList[i])): toWrite.append("NA")
    else: toWrite.append(str(dataPrList[i][j]))
  outputFile.write("\t".join(toWrite)+"\n")
outputFile.close()

for e in toRemove2: os.system("rm "+e)



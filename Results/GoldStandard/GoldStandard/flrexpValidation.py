
###################################################################################################
# Libraries
###################################################################################################

import os
import sys
import math
import operator
import numpy as np
from pickle import load
from optparse import OptionParser,BadOptionError,AmbiguousOptionError
from numpy import array, mean, median, std, log10
from scipy.stats import ks_2samp, ttest_ind, spearmanr, pearsonr

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

###################################################################################################
# Input Parameters
###################################################################################################

# Initialization of input parser
current_version = "0.0.1"

usage_message = ("\n--------------------------------------------------\n"
                 "This script performs the FLR-Exp evaluation procedure.\n\n"
                 "Usage: python flrexpValidation.py [options] <EXPRESSION_FILE> <MPBS_FILE1>\n"
                 "              <MPBS_FILE2> <FOOTPRINT_FILE1> <FOOTPRINT_FILE2>\n\n"
                 "Required Input:\n"
                 "<EXPRESSION_FILE>: A tab-separated table with the following columns (in order):\n"
                 "                   Name of transcription factors, expression fold change,\n"
                 "                   expression in cell 1 and expression in cell 2.\n"
                 "<MPBS_FILE1>: A bed file containing all motif-predicted binding sites (MPBSs).\n"
                 "              The value in the SCORE filed must reflect some quality measure of\n"
                 "              these MPBSs in cell 1.\n"
                 "<MPBS_FILE2>: A bed file containing all motif-predicted binding sites (MPBSs).\n"
                 "              The value in the SCORE filed must reflect some quality measure of\n"
                 "              these MPBSs in cell 2.\n"
                 "<FOOTPRINT_FILE1>: A bed file containing the footprint predictions in cell 1.\n"
                 "<FOOTPRINT_FILE2>: A bed file containing the footprint predictions in cell 2.\n"
                 "--------------------------------------------------")
version_message = "FLR-Exp evaluation. Version: "+str(current_version)
parser = PassThroughOptionParser(usage = usage_message, version = version_message)

# Optional Input Options
parser.add_option("--output-location", dest = "output_location", type = "string", metavar="PATH", 
                  default = os.getcwd(),
                  help = ("Path where the output files will be written."))

# Input variables
options, arguments = parser.parse_args()
if(not arguments or len(arguments) != 5):
    print("Wrong number of arguments!")
    sys.exit(0)
expFileName = arguments[0]
mpbsFileName1 = arguments[1]
mpbsFileName2 = arguments[2]
fpFileName1 = arguments[3]
fpFileName2 = arguments[4]
outputLocation = options.output_location
if(outputLocation[-1] != "/"): outputLocation+="/"

###################################################################################################
# Input Structures
###################################################################################################

# Labels
c1Name = expFileName.split("/")[-1].split(".")[0].split("_")[2]
c2Name = expFileName.split("/")[-1].split(".")[0].split("_")[3]
metrName = mpbsFileName1.split("/")[-1].split(".")[0].split("_")[-1]
fpName = fpFileName1.split("/")[-1].split(".")[0].split("_")[-1]

# Remove vectors
toRemove = []

# Extra parameters
myparams = load(open("./bin/binf","rb"))
metrName = myparams[0][metrName]
mTrans = myparams[1]
htd = myparams[2]
itv = myparams[3]
tcAdd = myparams[4]
pseudocount = myparams[5]
fptd = myparams[6]
cl = myparams[7]
c3Name = filter(lambda x: x not in [c1Name,c2Name], cl)[0]

###################################################################################################
# Reading Expression
###################################################################################################

expDict = dict()
expFile = open(expFileName,"r")
expFile.readline()
for line in expFile:
  ll = line.strip().split("\t")
  expDict[ll[0]] = [float(e) for e in ll[1:]]
expFile.close()
factorList = sorted(expDict.keys())

###################################################################################################
# Intersect MPBS / FP
###################################################################################################

toRemoveTemp = []

# Get FP
try: 
  newF1B = "./bin/"+fptd[c1Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bb"
  fpfn1 = "./bin/"+fptd[c1Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bed"
  os.system("./bin/bigBedToBed "+newF1B+" "+fpfn1)
  toRemoveTemp.append(fpfn1)
except Exception: fpfn1 = fpFileName1
try: 
  newF2B = "./bin/"+fptd[c2Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bb"
  fpfn2 = "./bin/"+fptd[c2Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bed"
  os.system("./bin/bigBedToBed "+newF2B+" "+fpfn2)
  toRemoveTemp.append(fpfn2)
except Exception: fpfn2 = fpFileName2
try: 
  newF3B = "./bin/"+fptd[c3Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bb"
  fpfn3 = "./bin/"+fptd[c3Name][htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][0]]+".bed"
  os.system("./bin/bigBedToBed "+newF3B+" "+fpfn3)
  toRemoveTemp.append(fpfn3)
except Exception: fpfn3 = fpFileName3

# Intersection 1
sortFileNameM = outputLocation+"sortM.bed"; toRemoveTemp.append(sortFileNameM)
sortFileNameF = outputLocation+"sortF.bed"; toRemoveTemp.append(sortFileNameF)
intFileName1 = outputLocation+"int1.bed"; toRemove.append(intFileName1)
os.system("sort -k1,1 -k2,2n "+mpbsFileName1+" > "+sortFileNameM)
os.system("sort -k1,1 -k2,2n "+fpfn1+" > "+sortFileNameF)
os.system("intersectBed -a "+sortFileNameM+" -b "+sortFileNameF+" -wa -u > "+intFileName1)

# Intersection 2
sortFileNameM = outputLocation+"sortM.bed"
sortFileNameF = outputLocation+"sortF.bed"
intFileName2 = outputLocation+"int2.bed"; toRemove.append(intFileName2)
os.system("sort -k1,1 -k2,2n "+mpbsFileName2+" > "+sortFileNameM)
os.system("sort -k1,1 -k2,2n "+fpfn2+" > "+sortFileNameF)
os.system("intersectBed -a "+sortFileNameM+" -b "+sortFileNameF+" -wa -u > "+intFileName2)

# Intersection 3
sortFileNameM = outputLocation+"sortM.bed"
sortFileNameF = outputLocation+"sortF.bed"
intFileName3 = outputLocation+"int3.bed"; toRemove.append(intFileName3)
os.system("sort -k1,1 -k2,2n "+fpfn3+" > "+sortFileNameF)
os.system("intersectBed -a "+sortFileNameM+" -b "+sortFileNameF+" -wa -u > "+intFileName3)

for e in toRemoveTemp: os.system("rm "+e)

###################################################################################################
# Fetching distribution
###################################################################################################

distDict1 = dict()
intFile1 = open(intFileName1,"r")
for line in intFile1:
  ll = line.strip().split("\t")
  if(ll[4] in ["-Inf","NaN"]): continue
  try: distDict1[ll[3]].append(float(ll[4]))
  except Exception: distDict1[ll[3]] = [float(ll[4])]
intFile1.close()

distDict2 = dict()
intFile2 = open(intFileName2,"r")
for line in intFile2:
  ll = line.strip().split("\t")
  if(ll[4] in ["-Inf","NaN"]): continue
  try: distDict2[ll[3]].append(float(ll[4]))
  except Exception: distDict2[ll[3]] = [float(ll[4])]
intFile2.close()

distDict3 = dict()
intFile3 = open(intFileName3,"r")
for line in intFile3:
  ll = line.strip().split("\t")
  if(ll[4] in ["-Inf","NaN"]): continue
  try: distDict3[ll[3]].append(float(ll[4]))
  except Exception: distDict3[ll[3]] = [float(ll[4])]
intFile3.close()

for k in factorList:
  singleVec = distDict1[k]+distDict2[k]+distDict3[k]
  minV = min(singleVec)
  maxV = max(singleVec)
  distDict1[k] = [(float(e)-minV)/(maxV-minV) for e in distDict1[k]]
  distDict2[k] = [(float(e)-minV)/(maxV-minV) for e in distDict2[k]]

###################################################################################################
# Normalize Signal (Filter Metrics)
###################################################################################################

# Evaluating min/max
minFc = float(min([expDict[e][0] for e in factorList]))
maxFc = float(max([expDict[e][0] for e in factorList]))

# Normalization
for k in factorList:
  xnorm = ((expDict[k][0] - minFc) / (maxFc - minFc))
  mf1 = itv[0] + ((itv[1]-itv[0]) * (1.0-xnorm))
  distDict1[k] = [float(e)*mf1 for e in distDict1[k]]
  mf2 = itv[0] + ((itv[1]-itv[0]) * (xnorm))
  distDict2[k] = [float(e)*mf2 for e in distDict2[k]]

###################################################################################################
# Create Multivar Table and FLR-Exp
###################################################################################################

# Initializations
ksDict = dict()
fcArray = []
ksArray = []

# TF Loop
for k in factorList:

  # Vectors
  vector1 = distDict1[k]
  vector2 = distDict2[k]
  extracount = htd[c1Name+"_"+c2Name][mTrans[fpName]+"_"+metrName][1]

  # KS Test
  try: ksStat, ksPv = ks_2samp(vector1, vector2)
  except Exception: kStat = 0.0; ksPv = 1.0
  if(expDict[k][0] < 0):
    ksStat = -ksStat + extracount
    ksPv = log10(ksPv+pseudocount)
  else:
    ksStat = ksStat - extracount
    ksPv = -log10(ksPv+pseudocount)
  
  ksDict[k] = [str(e) for e in [k,expDict[k][0],expDict[k][1],expDict[k][2],ksStat,ksPv]]
  fcArray.append(expDict[k][0])
  ksArray.append(ksStat)

# Evaluating correlation
fcArray = array(fcArray)
ksArray = array(ksArray)
coeff, pv = spearmanr(fcArray,ksArray)

for e in toRemove: os.system("rm "+e)

###################################################################################################
# Writing Results
###################################################################################################

headerVec = ["FACTOR","EXPR_FOLDCHANGE","EXPR_"+c1Name,"EXPR_"+c2Name,fpName+"_ks_stat",fpName+"_ks_pvalue"]
outputFileName = outputLocation+fpName+"_flrexp.txt"
outputFile = open(outputFileName,"w")
outputFile.write("\t".join(headerVec)+"\n")
for k in factorList: outputFile.write("\t".join(ksDict[k])+"\n")
corrVec = ["Spearman Correlation","","","",str(coeff),str(pv)]
outputFile.write("\t".join(corrVec)+"\n")
outputFile.close()



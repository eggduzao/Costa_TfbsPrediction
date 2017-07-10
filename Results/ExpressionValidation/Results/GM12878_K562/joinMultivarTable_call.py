
# Import
import os
import sys
from glob import glob

neg = "GM12878"
pos = "K562"
badTC = "Centipede_90"

###############################################################################
# Correcting TC tc -> TC fos
###############################################################################

# Error
inErrorFile = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/"+badTC+"_error.txt"
inF = open(inErrorFile,"r")
header = inF.readline()
table = [line.strip().split("\t") for line in inF]
newtable = [v[:2]+[v[3],v[3],v[4]] for v in table]
inF.close()
outF = open(inErrorFile,"w")
outF.write(header)
for v in newtable: outF.write("\t".join(v)+"\n")
outF.close()

# Rank
inErrorFile = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/"+badTC+"_rank.txt"
inF = open(inErrorFile,"r")
header = inF.readline()
table = [line.strip().split("\t") for line in inF]
newtable = [v[:6]+v[2:4]+v[8:14]+v[10:12] for v in table]
inF.close()
outF = open(inErrorFile,"w")
outF.write(header)
for v in newtable: outF.write("\t".join(v)+"\n")
outF.close()

# Table
inErrorFile = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/"+badTC+"_multivar_table.txt"
inF = open(inErrorFile,"r")
header = inF.readline()
table = [line.strip().split("\t") for line in inF]
newtable = [v[0:12]+v[20:28]+v[20:36] for v in table]
inF.close()
outF = open(inErrorFile,"w")
outF.write(header)
for v in newtable: outF.write("\t".join(v)+"\n")
outF.close()

###############################################################################
# Merging multivar table
###############################################################################

# Input
toRemove = []
inListTable = sorted(glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/*_multivar_table.txt"))
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/multivar_table.txt"

# Execution
headerF = inListTable[0]+"HEADER.txt"; toRemove.append(headerF)
os.system("cut -f 1,2,3,4 "+inListTable[0]+" > "+headerF)
for inFName in inListTable:
  fn = inFName+"TEMP.txt"; toRemove.append(fn)
  os.system("cut -f "+",".join([str(e) for e in range(5,37)])+" "+inFName+" > "+fn)
os.system("paste "+" ".join(toRemove)+" > "+outFileName)
for e in toRemove: os.system("rm "+e)

###############################################################################
# Merging rank
###############################################################################

# Input
inListRank = glob("/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/*_rank.txt")
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/rank.txt"

# Execution
tableVec = [[],[],[],[],[],[],[],[]]
for inFName in inListRank:
  inF = open(inFName,"r") 
  header = inF.readline()
  ll = inF.readline().strip().split("\t")
  for i in range(0,len(tableVec)):
    tableVec[i].append([ll[2*i],float(ll[(2*i)+1])])
  inF.close()
for i in range(0,len(tableVec)): tableVec[i] = sorted(tableVec[i],key=lambda x:x[1],reverse=True)
outFile = open(outFileName,"w")
outFile.write(header)
for i in range(0,len(tableVec[0])):
  vec = []
  for j in range(0,len(tableVec)):
    vec.append(tableVec[j][i][0])
    vec.append(str(tableVec[j][i][1]))
  outFile.write("\t".join(vec)+"\n")
outFile.close()

###############################################################################
# Merging error
###############################################################################

# Input
inLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/"
inExt = "*_error.txt"
outFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/error.txt"

# Execution
os.system("head -n 1 /work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ExpressionValidation/Results/"+neg+"_"+pos+"/multivar_table/Boyle_error.txt > "+outFileName)
os.system("find "+inLoc+" -name \""+inExt+"\" | xargs -n 1 tail -n +2 >> "+outFileName)



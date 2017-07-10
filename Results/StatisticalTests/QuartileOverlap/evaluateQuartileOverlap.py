
import os
import sys

inFile = open("./quartile.txt","r")
methodVec = inFile.readline().strip().split("\t")
resDict = dict([(e,[]) for e in methodVec])
for line in inFile:
  ll = line.strip().split("\t")
  for i in range(0,len(ll)):
    resDict[methodVec[i]].append(ll[i])
inFile.close()

selList = ["HINT_BC_AUC_10","HINT_BCN_AUC_10","HINT_AUC_10","DNase2TF_AUC_10","PIQ_AUC_10"]
vecList = []
nb = int(0.25*len(resDict["HINT_BC_AUC_10"] ))
for m in selList:
  vec = resDict[m] 
  vecList.append(set(vec[:nb]))

#print vecList
print set.intersection(vecList[0],vecList[3],vecList[4])

print 100.0*float(len(set.intersection(vecList[0],vecList[3],vecList[4])))/float(nb)


  


##################################################
## Statistics on Footprints
##################################################

# Import
import os
import sys
import aux
import glob
import Table
import numpy as np

# Input & output
cellLine = sys.argv[1]
evName = sys.argv[2]
footLocation = sys.argv[3]
outputFileName = sys.argv[4]

# Additional parameters
footLoc = footLocation+cellLine+"/"+evName+"/"
outputLocation = "/".join(outputFileName.split("/")[:-1])+"/"
headerVec = ["Factor","TP","FP","TN","FN","Acc","BAc","Sens","Spec","PPV","NPV"]
TP = 0; FP = 1; TN = 2; FN = 3

# Iterating on result files
dataMatrixList = []
captionList = []
for footFileName in glob.glob(footLoc+"*.bed"):

    # Initializing structures
    resDict = dict()

    # Threshold
    thresh = 1001.0
    if("Pique" in footFileName): thresh = 0.99

    # Evaluating caption
    captionList.append(" ".join(footFileName.split("/")[-1].split(".")[0].split("_"))+" "+str(thresh))

    # Iterating on file
    footFile = open(footFileName,"r")
    for line in footFile:

        # Initialization
        ll = line.strip().split("\t")
        lll = ll[3].split(":")
        currFacName = lll[0]
        ev = lll[1]
        score = float(ll[4])
        try: resDict[currFacName]
        except Exception: resDict[currFacName] = [0.0,0.0,0.0,0.0]

        # Contingency Table
        if(ev == "Y" and score >= thresh): resDict[currFacName][TP] += 1.0
        elif(ev == "Y" and score < thresh): resDict[currFacName][FN] += 1.0
        elif(ev == "N" and score >= thresh): resDict[currFacName][FP] += 1.0
        elif(ev == "N" and score < thresh): resDict[currFacName][TN] += 1.0
        elif(ev == "." and score >= thresh): resDict[currFacName][FP] += 1.0
        elif(ev == "." and score < thresh): resDict[currFacName][TN] += 1.0

    footFile.close()

    # Creating dataMatrix
    dataMatrix = []
    factorList = resDict.keys()
    factorList.sort()
    for factor in factorList:
        dataVec = [factor]
        tp = resDict[factor][TP]; fp = resDict[factor][FP]; tn = resDict[factor][TN]; fn = resDict[factor][FN]
        dataVec.append(int(tp)); dataVec.append(int(fp)); dataVec.append(int(tn)); dataVec.append(int(fn))
        if((tp+tn+fp+fn) > 0.0): dataVec.append(round((tp+tn)/(tp+tn+fp+fn),4))
        else: dataVec.append(0.0)
        if((tp+fn) > 0.0 and (tn+fp) > 0.0): dataVec.append(round( ((tp/(tp+fn))+(tn/(tn+fp)))/2.0 ,4))
        else: dataVec.append(0.0)
        if((tp+fn) > 0.0): dataVec.append(round(tp/(tp+fn),4))
        else: dataVec.append(0.0)
        if((tn+fp) > 0.0): dataVec.append(round(tn/(tn+fp),4))
        else: dataVec.append(0.0)
        if((tp+fp) > 0.0): dataVec.append(round(tp/(tp+fp),4))
        else: dataVec.append(0.0)
        if((tn+fn) > 0.0): dataVec.append(round(tn/(tn+fn),4))
        else: dataVec.append(0.0)
        dataMatrix.append(dataVec)

    # Updating dataMatrixList
    dataMatrix = [list(e) for e in np.array(dataMatrix).T]
    dataMatrixList.append(dataMatrix)

# Creating table
aux.createTable(captionList,headerVec,dataMatrixList,outputFileName)



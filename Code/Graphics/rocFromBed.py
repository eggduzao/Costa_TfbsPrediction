#################################################################################################
# Creates a ROC curve from a bed file already separated by evidence. True positives are marked
# with Y in the NAME field and false positives are marked with N.
# This differs from the regular rocFromBed because it takes into account the fact that the method
# is site-centric (SC) or segmentation-based (SB).
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsName = Name of the MPBS. The same name as in the NAME field.")
params.append("    2. typeList = List of file types (comma separated): SB or SC")
params.append("    3. labelList = List of labels to each of the bed files in the list.")
params.append("    4. bedList = List of files containing bed coordinates separated by comma.")
params.append("    5. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <mpbsName>.png = ROC graph.")
params.append("    2. <mpbsName>_stats.txt = AUC, sensitivity and specificity.")
params.append("    4. <mpbsName>_roc.txt = ROC in txt format.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
import operator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.integrate import simps, trapz
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsName = sys.argv[1]
typeList = sys.argv[2].split(",")
labelList = sys.argv[3].split(",")
bedList = sys.argv[4].split(",")
outputLocation = sys.argv[5]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Standardize function
def standardize(vec):
    maxN = max(vec)
    minN = min(vec)
    return [(e-minN)/(maxN-minN) for e in vec]

# Parameters
maxPoints = 1000
fpr_auc = 0.1
fpr_auc2 = 0.01
cellType = outputLocation.split("/")[-2]

#############################################################
# Calculating ROC
#############################################################

# Reading data points
dataXList = []; dataYList = []; dataRcList = []; dataPrList = []
dataXListFprSb = []; dataYListFprSb = [];
newLabelList = []
for k in range(0,len(bedList)):

    # Initialization
    bedFileName = bedList[k]
    if(not os.path.isfile(bedFileName)): continue
    countX = 0; countY = 0
    dataX = [countX]; dataY = [countY]
    dataPr = [0.0]; dataRc = [0.0]

    # Separating MPBSs
    sepFileName = outputLocation+bedFileName.split("/")[-1].split(".")[0]+"_"+mpbsName+"_sep.bed"
    bedFile = open(bedFileName,"r")
    sepFile = open(sepFileName,"w")
    counter = 0
    for line in bedFile:
        if(line.strip().split("\t")[3].split(":")[0] != mpbsName): continue
        sepFile.write(line)
        counter += 1
    bedFile.close()
    sepFile.close()
    if(counter == 0): continue
    newLabelList.append(labelList[k])

    # Sorting in decreasing order
    sortedFileName = outputLocation+bedFileName.split("/")[-1].split(".")[0]+"_"+mpbsName+"_sorted.bed"
    os.system("sort -r -k5 -g "+sepFileName+" > "+sortedFileName)

    # Evaluating ROC points
    bedFile = open(sortedFileName,"r")
    countX_fprSb = 0.0; countY_fprSb = 0.0; flagFirst = True
    for line in bedFile:
        ll = line.strip().split("\t")
        lll = ll[3].split(":")
        if(lll[0] != mpbsName): continue
        if(lll[1] == "Y"):
            countY += 1
            dataX.append(countX); dataY.append(countY)
            dataPr.append(float(countY)/(countX+countY))
            dataRc.append(countY)
            if(typeList[k] == "SB" and len(lll) > 2 and flagFirst): countY_fprSb += 1.0
            else: flagFirst = False
        else:
            countX += 1
            dataX.append(countX); dataY.append(countY)
            dataPr.append(float(countY)/(countX+countY))
            dataRc.append(countY)
            if(typeList[k] == "SB" and len(lll) > 2 and flagFirst): countX_fprSb += 1.0
            else: flagFirst = False
    dataXList.append([e*(1.0/countX) for e in dataX])
    dataYList.append([e*(1.0/countY) for e in dataY])
    dataRc = [e*(1.0/countY) for e in dataRc]
    dataPr.append(0.0); dataRc.append(1.0)
    dataPrList.append(dataPr)
    dataRcList.append(dataRc)
    if(typeList[k] == "SB"):
        fpr_sb = countX_fprSb / countX
        tpr_sb = countY_fprSb / countY
        dataXListFprSb.append(fpr_sb)
        dataYListFprSb.append(tpr_sb)
    bedFile.close()

    # Termination
    os.remove(sepFileName)
    os.remove(sortedFileName)

# Evaluating AUC
statsList = [] # For each method: [AUC,AUC_10%,SENS_10%,SPEC_10%,AUC_M1,SENS_M1,SPEC_M1,AUC_M2,SENS_M2,SPEC_M2,...]
statsListPr = [] # For each method: [AUPR]
for i in range(0,len(dataXList)):

    # Initialization 
    vec = []
    vecPr = []

    # Evaluating regular AUC
    vec.append(auc(dataXList[i],dataYList[i]))
    vecPr.append(abs(trapz(dataRcList[i],dataPrList[i])))

    # Evaluating 10% FPR AUC
    fprVecX = []; fprVecY = []
    for j in range(0,len(dataXList[i])):
        if(dataXList[i][j] > fpr_auc): break
        fprVecX.append(dataXList[i][j]); fprVecY.append(dataYList[i][j]); 
    fprVecX.append(fpr_auc); fprVecY.append(fprVecY[-1])
    vec.append(auc(standardize(fprVecX),fprVecY))
    vec.append(fprVecY[-1])
    vec.append(1.0-fprVecX[-1])

    # Evaluating 1% FPR AUC
    fprVecX2 = []; fprVecY2 = []
    for j in range(0,len(dataXList[i])):
        if(dataXList[i][j] > fpr_auc2): break
        fprVecX2.append(dataXList[i][j]); fprVecY2.append(dataYList[i][j]); 
    fprVecX2.append(fpr_auc2); fprVecY2.append(fprVecY2[-1])
    vec.append(auc(standardize(fprVecX2),fprVecY2))
    vec.append(fprVecY2[-1])
    vec.append(1.0-fprVecX2[-1])

    # Evaluating SB FPR AUC
    for fpr_sb in dataXListFprSb:
        fprVecX = []; fprVecY = []
        for j in range(0,len(dataXList[i])):
            if(dataXList[i][j] > fpr_sb): break
            fprVecX.append(dataXList[i][j]); fprVecY.append(dataYList[i][j]);
        fprVecX.append(fpr_sb); fprVecY.append(fprVecY[-1])
        vec.append(auc(standardize(fprVecX),fprVecY))
        vec.append(fprVecY[-1])
        vec.append(1.0-fprVecX[-1])

    statsList.append(vec)
    statsListPr.append(vecPr)


#############################################################
# Writing Results
#############################################################

# Statistics Header
header = ["METHOD","AUC","AUC_FPR_10","SN_FPR_10","SP_FPR_10","AUC_FPR_1","SN_FPR_1","SP_FPR_1"]
sbLabelMethodList = [newLabelList[k] for k in range(0,len(newLabelList)) if typeList[k] == "SB"]
for e in sbLabelMethodList:
    header.append("AUC_"+e); header.append("SN_"+e); header.append("SP_"+e)

# Writing Statistics
outputFile = open(outputLocation+mpbsName+"_stats.txt","w")
outputFile.write("\t".join(header)+"\n")
for i in range(0,len(statsList)): outputFile.write("\t".join([newLabelList[i]]+[str(e) for e in statsList[i]])+"\n")
outputFile.close()

# Writing Statistics AUPR
outputFile = open(outputLocation+mpbsName+"_statsPR.txt","w")
headerPr = ["METHOD","AUC"]
outputFile.write("\t".join(headerPr)+"\n")
for i in range(0,len(statsListPr)): outputFile.write("\t".join([newLabelList[i]]+[str(e) for e in statsListPr[i]])+"\n")
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
for i in range(0,len(dataPrList)):
    newDataPrList = []; newDataRcList = []
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
outputFile = open(outputLocation+mpbsName+"_roc.txt","w")
graphLabelList = []
for e in newLabelList:
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
outputFile = open(outputLocation+mpbsName+"_prc.txt","w")
graphLabelList = []
for e in newLabelList:
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

#############################################################
# Creating Line Graph AUC
#############################################################

# Creating figure
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

# Colors
# Black        / Darker blue  / Dark green    / Darker red  / Darker purple  / Darker pink  / Darker Grey  / Darker Yellow  
# Dark Orange  / Dark blue    / Med green     / Dark red    / Dark purple    / Dark pink    / Dark grey    / Dark yellow 
# Light Orange / Med blue     / Med green     / Med red     / Med purple     / Med pink     / Med grey     / Med yellow
# Dark Brown   / Light blue   / Light green   / Light red   / Light purple   / Light pink   / Light grey   / Light yellow
# Light Brown  / Lighter blue / Lighter green / Lighter red / Lighter purple / Lighter pink / Lighter grey / Lighter yellow
colorList = ["#000000", "#000099", "#006600", "#990000", "#660099", "#CC00CC", "#222222", "#CC9900", 
             "#FF6600", "#0000CC", "#336633", "#CC0000", "#6600CC", "#FF00FF", "#555555", "#CCCC00", 
             "#FF9900", "#0000FF", "#33CC33", "#FF0000", "#663399", "#FF33FF", "#888888", "#FFCC00",
             "#663300", "#009999", "#66CC66", "#FF3333", "#9933FF", "#FF66FF", "#AAAAAA", "#FFCC33",
             "#993300", "#00FFFF", "#99FF33", "#FF6666", "#CC99FF", "#FF99FF", "#CCCCCC", "#FFFF00"]
colorList += colorList+colorList

# Plot
for i in range(0,len(dataXList)):
    ax.plot(dataXList[i], dataYList[i], color=colorList[i], label=newLabelList[i]+": "+str(round(statsList[i][0],4)))

# Plot red diagonal line
ax.plot([0,1.0], [0,1.0], color="#CCCCCC", linestyle="--", alpha=1.0)

# Plotting SB points
#sbIndexList = [k for k in range(0,len(newLabelList)) if typeList[k] == "SB"]
#for i in range(0,len(sbIndexList)):
#    plt.plot([1.0-statsList[sbIndexList[i]][6+(i*3)]], [statsList[sbIndexList[i]][5+(i*3)]], 'o', color=colorList[sbIndexList[i]])

# Line legend
leg = ax.legend(bbox_to_anchor=(1.0, 0.0, 1.0, 1.0), loc=2, ncol=2, borderaxespad=0., title="AUC", prop={'size':4})
for e in leg.legendHandles: e.set_linewidth(2.0)

# Titles and Axis Labels
ax.set_title(mpbsName)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")

# Ticks
slots = 10.0
ax.grid(True, which='both')
ax.set_xticks(np.arange(0.0,1.01,0.1))
ax.set_yticks(np.arange(0.0,1.01,0.1))
for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(10)
for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(10)

# Axis limits
pylab.xlim([0,1.0])
pylab.ylim([0,1.0])

# Saving figure
fig.savefig(outputLocation+mpbsName+"_ROC.png", format="png", dpi=300, bbox_inches='tight', bbox_extra_artists=[leg]) 

#############################################################
# Creating Line Graph PRC
#############################################################

# Creating figure
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

# Colors
colorList = ["#000000", "#000099", "#006600", "#990000", "#660099", "#CC00CC", "#222222", "#CC9900", 
             "#FF6600", "#0000CC", "#336633", "#CC0000", "#6600CC", "#FF00FF", "#555555", "#CCCC00", 
             "#FF9900", "#0000FF", "#33CC33", "#FF0000", "#663399", "#FF33FF", "#888888", "#FFCC00",
             "#663300", "#009999", "#66CC66", "#FF3333", "#9933FF", "#FF66FF", "#AAAAAA", "#FFCC33",
             "#993300", "#00FFFF", "#99FF33", "#FF6666", "#CC99FF", "#FF99FF", "#CCCCCC", "#FFFF00"]
colorList += colorList+colorList

# Plot
for i in range(0,len(dataPrList)):
    ax.plot(dataRcList[i], dataPrList[i], color=colorList[i], label=newLabelList[i]+": "+str(round(statsListPr[i][0],4)))

# Plot red diagonal line
ax.plot([0,1.0], [0,1.0], color="#CCCCCC", linestyle="--", alpha=1.0)

# Line legend
leg = ax.legend(bbox_to_anchor=(1.0, 0.0, 1.0, 1.0), loc=2, ncol=2, borderaxespad=0., title="AUC", prop={'size':4})
for e in leg.legendHandles: e.set_linewidth(2.0)

# Titles and Axis Labels
ax.set_title(mpbsName)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

# Ticks
slots = 10.0
ax.grid(True, which='both')
ax.set_xticks(np.arange(0.0,1.01,0.1))
ax.set_yticks(np.arange(0.0,1.01,0.1))
for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(10)
for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(10)

# Axis limits
pylab.xlim([0,1.0])
pylab.ylim([0,1.0])

# Saving figure
fig.savefig(outputLocation+mpbsName+"_PRC.png", format="png", dpi=300, bbox_inches='tight', bbox_extra_artists=[leg]) 



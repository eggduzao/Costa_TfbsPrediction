#################################################################################################
# Creates a histogram of the positions of the motifs relative to the center of the predictions
# made.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. normalizeVec = If 'y' then normalize the plot vectors.")
params.append("    2. windowLen = Total length of the window surrounding predictions.")
params.append("    3. mpbsName = Name of the MPBS. Put 'all' to consider all motifs.")
params.append("    4. mpbsFileName = Name of the MPBS file.")
params.append("    5. predLabelList = List of prediction labels.")
params.append("    6. predFileNameList = List of prediction file names.")
params.append("    7. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <graphName>_<line|box>.<png|txt> = Graphs in png and txt formats.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
import pylab
import numpy as np
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
normalizeVec = sys.argv[1]
windowLen = int(sys.argv[2])
mpbsName = sys.argv[3]
mpbsFileName = sys.argv[4]
predLabelList = sys.argv[5].split(",")
predFileNameList = sys.argv[6].split(",")
outputLocation = sys.argv[7]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Parameters
it = 10

#############################################################
# Calculating Intersection
#############################################################

# Fetching true mpbs from file
mpbsFileLastName = mpbsFileName.split("/")[-1].split(".")[0]
mpbsFile = open(mpbsFileName,"r")
mpbsTempFileName = outputLocation+mpbsName+"_mpbstemp.bed"
mpbsTempFile = open(mpbsTempFileName,"w")
for line in mpbsFile:
    if(mpbsName == "all" and ":Y" in line): mpbsTempFile.write(line)
    elif(mpbsName in line and ":Y" in line): mpbsTempFile.write(line)
mpbsFile.close()
mpbsTempFile.close()

# Iterating on prediction files
plotMatrix = []
boxMatrix = []
for predFileName in predFileNameList:

    # Cutting predictions file
    predFile = open(predFileName,"r")
    predTempFileName1 = outputLocation+mpbsName+"_predtemp.bed"
    predTempFile = open(predTempFileName1,"w")
    for line in predFile:
        ll = line.strip().split("\t")
        # if(ll[0] == "chr1"): continue # XXX Removing chr1
        predTempFile.write("\t".join(ll[0:3])+"\n")
    predFile.close()
    predTempFile.close()
    predTempSortedFileName = outputLocation+mpbsName+"_predtempsort.bed"
    os.system("sort -k1,1 -k2,2g "+predTempFileName1+" > "+predTempSortedFileName)

    # Merging predictions
    #predTempFileName2 = outputLocation+mpbsName+"_predtempmerge.bed"
    #os.system("mergeBed -i "+predTempSortedFileName+" > "+predTempFileName2)

    # Intersection
    intFileName1 = outputLocation+mpbsName+"_intersect1.bed"
    os.system("intersectBed -a "+predTempSortedFileName+" -b "+mpbsTempFileName+" -wa -wb > "+intFileName1)

    # Processing intersection - Taking the FIRST prediction that overlapped same motif
    #intFile1 = open(intFileName1,"r")
    #intFileName2 = outputLocation+mpbsName+"_intersect2.bed"
    #intFile2 = open(intFileName2,"w")
    #prevLine = ""
    #for line in intFile1:
    #    ll = line.strip().split("\t")
    #    currLine = "\t".join(ll[3:10])
    #    if(currLine == prevLine): continue
    #    prevLine = currLine
    #    intFile2.write(line)
    #intFile1.close()
    #intFile2.close()

    # Processing intersection - Calculating the relative positions
    intFile = open(intFileName1,"r")
    intDict = dict()
    for line in intFile:
        ll = line.strip().split("\t")
        p1 = int(ll[1]); p2 = int(ll[2]); m1 = int(ll[4]); m2 = int(ll[5])
        mid = (p1+p2)/2
        p1_ext = mid - (windowLen/2); p2_ext = mid + (windowLen/2)
        a = m1-p1_ext; b = m2-p1_ext
        currKey = ll[3]+":"+ll[4]+":"+ll[5]
        if(currKey not in intDict): intDict[currKey] = []
        counter = 0.0; currVec = []; currSumm = 0.0
        for i in range(a,b):
            if(i >= 0 and i < windowLen):
                currVec.append(i)
                currSumm += i-(windowLen/2)
                counter += 1.0
        if(counter > 0.0): currSumm /= counter
        intDict[currKey].append([currVec,currSumm])
    intFile.close()

    # Reading dictionary
    plotVec = [0.0] * windowLen
    boxVec = [];
    for k in intDict.keys():
        if(not intDict[k]): continue
        avgVec = [abs(e[1]) for e in intDict[k]]
        minIndex = avgVec.index(min(avgVec))
        try:
            boxVec.append(intDict[k][minIndex][1])
        except Exception:
            print avgVec
            print minIndex
            for e in intDict[k]: print e
        for e in intDict[k][minIndex][0]: plotVec[e] += 1.0

    # Normalizing vector
    if(normalizeVec == "y"):
        minV = min(plotVec)
        maxV = max(plotVec)
        plotVec = [(e-minV)/(maxV-minV) for e in plotVec]
    plotMatrix.append(plotVec)
    boxMatrix.append(boxVec)

    # Removing temp files
    os.system("rm "+predTempFileName1+" "+intFileName1+" "+predTempSortedFileName)

# Removing temp files
os.system("rm "+mpbsTempFileName)

#############################################################
# Writing Plots
#############################################################

# Deciding graph name
if(mpbsName == "all"): graphName = mpbsFileLastName
else: graphName = mpbsName

# Writing line graph
outputFile = open(outputLocation+graphName+"_line.txt","w")
outputFile.write("\t".join(["X"]+predLabelList)+"\n")
indexList = range(int(-math.floor(windowLen/2.0)),int(math.ceil(windowLen/2.0))+1)
for j in range(0,len(plotMatrix[0])):
    toWrite = [str(indexList[j])]
    for i in range(0,len(plotMatrix)): toWrite.append(str(plotMatrix[i][j]))
    outputFile.write("\t".join(toWrite)+"\n")
outputFile.close()

# writing boxplot graph
outputFile = open(outputLocation+graphName+"_box.txt","w")
outputFile.write("\t".join(predLabelList)+"\n")
lenVec = [len(e) for e in boxMatrix]
maxInd = lenVec.index(max(lenVec))
for j in range(0,len(boxMatrix[maxInd])):
    toWrite = []
    for i in range(0,len(boxMatrix)):
        if(j >= len(boxMatrix[i])): toWrite.append("NA")
        else: toWrite.append(str(boxMatrix[i][j]))
    outputFile.write("\t".join(toWrite)+"\n")
outputFile.close()

#############################################################
# Line Graph
#############################################################

# Creating figure
fig = plt.figure(figsize=(8,5), facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

# Colors
# Black        / Darker blue  / Dark green  / Darker red  / Darker purple  / Darker pink  / Darker Grey  / Darker Yellow  
# Dark Orange  / Dark blue    / Med green   / Dark red    / Dark purple    / Dark pink    / Dark grey    / Dark yellow 
# Light Orange / Med blue     / Lighe green / Med red     / Med purple     / Med pink     / Med grey     / Med yellow
# Dark Brown   / Light blue   / Lighe green / Light red   / Light purple   / Light pink   / Light grey   / Light yellow
# Light Brown  / Lighter blue / Lighe green / Lighter red / Lighter purple / Lighter pink / Lighter grey / Lighter yellow
colorList = ["#000000", "#000099", "#006600", "#990000", "#660099", "#CC00CC", "#222222", "#CC9900", 
             "#FF6600", "#0000CC", "#336633", "#CC0000", "#6600CC", "#FF00FF", "#555555", "#CCCC00", 
             "#FF9900", "#0000FF", "#33CC33", "#FF0000", "#663399", "#FF33FF", "#888888", "#FFCC00",
             "#663300", "#009999", "#66CC66", "#FF3333", "#9933FF", "#FF66FF", "#AAAAAA", "#FFCC33",
             "#993300", "#00FFFF", "#99FF33", "#FF6666", "#CC99FF", "#FF99FF", "#CCCCCC", "#FFFF00"]
colorList += colorList+colorList

# Plot
for i in range(0,len(plotMatrix)):
    ax.plot(range(0,len(plotMatrix[i])), plotMatrix[i], color=colorList[i], label=predLabelList[i])

# Line legend
leg = ax.legend(bbox_to_anchor=(1.0, 0.0, 1.0, 1.0), loc=2, ncol=1, borderaxespad=0., title="Legend")
for e in leg.legendHandles: e.set_linewidth(3.0)

# Titles and Axis Labels
ax.set_title(graphName)
ax.set_ylabel("Number of True Motifs Found")

# Ticks
ax.set_xticks(range(0,windowLen+1,windowLen/it))
ax.set_xticklabels(range(int(-math.floor(windowLen/2.0)),int(math.ceil(windowLen/2.0))+1,windowLen/it))
if(normalizeVec == "y"): ax.set_yticks(np.arange(0.0,1.1,0.1))
ax.grid(True, which='both')
for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(8)
for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(8)

# Axis limits
pylab.xlim([0.0,windowLen])
if(normalizeVec == "y"): pylab.ylim([0.0,1.03])

# Saving figure
fig.savefig(outputLocation+graphName+"_line.png", format="png", dpi=300, bbox_inches='tight', bbox_extra_artists=[leg]) 


#############################################################
# Box Plot
#############################################################

# Creating figure
fig = plt.figure(figsize=(8,3), facecolor='w', edgecolor='k')
mpl.rcParams.update({'font.size': 8})
ax = fig.add_subplot(111)

# Plot
bp = ax.boxplot(boxMatrix, sym='k+', positions=range(1,len(boxMatrix)+1))
plt.setp(bp['whiskers'], color='k',  linestyle='-' )
plt.setp(bp['fliers'], markersize=3.0)

# Titles and Axis Labels
ax.set_title(graphName)
ax.set_ylabel("Motif Positions")

# Ticks
ax.set_xticks(range(1,len(boxMatrix)+1))
ax.set_xticklabels(predLabelList)
ax.set_yticks(range(int(-math.floor(windowLen/2.0)),int(math.ceil(windowLen/2.0))+1,windowLen/it))
ax.grid(True, which='both')
for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(4)
for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(8)

# Rotate labels
labels = ax.get_xticklabels()
for label in labels: label.set_rotation(90)

# Axis limits
pylab.ylim([-(windowLen/2.0),(windowLen/2.0)])

# Saving figure
fig.savefig(outputLocation+graphName+"_box.png", format="png", dpi=300, bbox_inches='tight') 



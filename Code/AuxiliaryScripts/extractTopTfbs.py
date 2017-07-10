#################################################################################################
# Extracts the top scored PWMs in two different files according to the following situations:
# 1. PWMs that have TFBS evidence.
# 2. PWMs that do not have TFBS evidence.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. topN = Number of top regions to retrieve.")
params.append("    2. pwmLocation = File containing the motif matched PWMs.")
params.append("    3. tfbsLocation = File containing the TFBS evidence.")
params.append("    4. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. top<topN>_1T.bed = PWMs that have TFBS evidence.")
params.append("    2. top<topN>_0T.bed = PWMs that do not have TFBS evidence.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import operator
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
topN = int(sys.argv[1])
pwmLocation = sys.argv[2]
tfbsLocation = sys.argv[3]
outputLocation = sys.argv[4]
if(outputLocation[-1] != "/"): outputLocation+="/"

#########################################
##### Initializing Structures
#########################################

# Initializing chromSet
chromSet = constants.getChromList()

# Initializing dictionaries (that will contain the list of intervals expressed by tuples)
pwmDict = aux.createBedDictFromMultipleFile(pwmLocation, features=[1,2,3,4,5], separator=" ")
tfbsDict = aux.createBedDictFromMultipleFile(tfbsLocation, features=[1,2], separator=" ")

# Force some keys to happen in the dictionary
for e in chromSet:
    if(e not in pwmDict.keys()): pwmDict[e] = []
    if(e not in tfbsDict.keys()): tfbsDict[e] = []

# Sorting dictionaries
pwmDict = gsort.sortBedDictionaries([pwmDict])[0]
tfbsDict = gsort.sortBedDictionaries([tfbsDict])[0]

#########################################
##### Separating PWM per Evidence Status
#########################################

# Initializing result lists
top_1T = []
top_0T = []

# Iterating on chromSet to separate PWMs by evidence status
for chrom in chromSet:

    # Disallowing chrY
    if(chrom == "chrY"): continue

    # Initializing lists
    pwmList = pwmDict[chrom]
    tfbsList = tfbsDict[chrom]

    # Checking the overlap between pwmList and tfbsList
    overlaps = []
    counts = 0
    for interval in pwmList:
        interval = interval[0:2]
        didBreak = False
        while(counts < len(tfbsList)):
            check = aux.checkTuplesOverlap(interval,tfbsList[counts])
            if(check == 0): 
                overlaps.append(True)
                didBreak = True
                break
            elif(check == -1):
                overlaps.append(False)
                didBreak = True
                break
            counts += 1
        if(not didBreak): overlaps.append(False)

    # Iterating on the overlaps list
    for i in range(0,len(overlaps)):
        if(overlaps[i]): top_1T.append([chrom] + pwmList[i])
        else: top_0T.append([chrom] + pwmList[i])

# End for chrom in chromSet

# Sorting result lists by the motif matching score
top_1T = sorted(top_1T,key = lambda k: k[4])
top_0T = sorted(top_0T,key = lambda k: k[4])


#########################################
##### Writing Results
#########################################

# File handling
outputFile_1T = open(outputLocation+"top"+str(min(len(top_1T),topN))+"_1T.bed","w")
outputFile_0T = open(outputLocation+"top"+str(min(len(top_0T),topN))+"_0T.bed","w")

# Writing results
topList = [top_1T, top_0T]
fileList = [outputFile_1T, outputFile_0T]
for i in range(0,len(topList)):
    for j in range(len(topList[i])-1,max(len(topList[i])-topN-1,-1),-1):
        fileList[i].write(topList[i][j][0]+" "+str(topList[i][j][1])+" "+str(topList[i][j][2])+" "+topList[i][j][3]+" "+str(topList[i][j][4])+" "+topList[i][j][5]+"\n")

# Termination
outputFile_1T.close()
outputFile_0T.close()



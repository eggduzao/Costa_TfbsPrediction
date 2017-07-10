#################################################################################################
# Extracts the top scored PWMs in four different files according to the following situations:
# 1. PWMs that have both TFBS and Footprint evidence.
# 2. PWMs that have TFBS evidence but not Footprint.
# 3. PWMs that does not have TFBS evidence but have Footprint.
# 4. PWMs that does not have neither TFBS nor Footprint evidence.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. topN = Number of top regions to retrieve.")
params.append("    2. separator = The separator of the footprint file. Can be \"tab\" or \"spc\".")
params.append("    3. pwmLocation = File containing the motif matched PWMs.")
params.append("    4. tfbsLocation = File containing the TFBS evidence.")
params.append("    5. footFileName = File containing the footprint evidence.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. top<topN>_1T_1F.bed = PWMs that have both TFBS and Footprint evidence.")
params.append("    2. top<topN>_1T_0F.bed = PWMs that have TFBS evidence but not Footprint.")
params.append("    3. top<topN>_0T_1F.bed = PWMs that dont have TFBS evidence but have Foot.")
params.append("    4. top<topN>_0T_0F.bed = PWMs that dont have neither TFBS nor Footprint.")
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
separator = sys.argv[2]
if(separator == "tab"): separator = "\t"
else: separator = " "
pwmLocation = sys.argv[3]
tfbsLocation = sys.argv[4]
footFileName = sys.argv[5]
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"

#########################################
##### Initializing Structures
#########################################

# Initializing chromSet
chromSet = constants.getChromList()

# Initializing dictionaries (that will contain the list of intervals expressed by tuples)
pwmDict = aux.createBedDictFromMultipleFile(pwmLocation, features=[1,2,3,4,5], separator=" ")
tfbsDict = aux.createBedDictFromMultipleFile(tfbsLocation, features=[1,2], separator=" ")
footDict = aux.createBedDictFromSingleFile(footFileName, features=[1,2], separator=separator)

# Force some keys to happen in the dictionary
for e in chromSet:
    if(e not in pwmDict.keys()): pwmDict[e] = []
    if(e not in tfbsDict.keys()): tfbsDict[e] = []
    if(e not in footDict.keys()): footDict[e] = []

# Sorting dictionaries
pwmDict = gsort.sortBedDictionaries([pwmDict])[0]
tfbsDict = gsort.sortBedDictionaries([tfbsDict])[0]
footDict = gsort.sortBedDictionaries([footDict])[0]


#########################################
##### Separating PWM per Evidence Status
#########################################

# Initializing result lists
top_1T_1F = []
top_1T_0F = []
top_0T_1F = []
top_0T_0F = []

# Iterating on chromSet to separate PWMs by evidence status
for chrom in chromSet:

    # Disallowing chrY
    if(chrom == "chrY"): continue

    # Initializing lists
    pwmList = pwmDict[chrom]
    lists = [tfbsDict[chrom],footDict[chrom]]

    # Checking the overlap with each element of PWM list and the elements of the other lists
    overlaps = [[],[]]
    counts = [0,0]
    for interval in pwmList:
        interval = interval[0:2]
        for i in range(0,len(lists)):
            didBreak = False
            while(counts[i] < len(lists[i])):
                check = aux.checkTuplesOverlap(interval,lists[i][counts[i]])
                if(check == 0): 
                    overlaps[i].append(True)
                    didBreak = True
                    break
                elif(check == -1):
                    overlaps[i].append(False)
                    didBreak = True
                    break
                counts[i] += 1
            if(not didBreak): overlaps[i].append(False)

    # Iterating on the overlap lists
    # overlaps[0] = Overlap between PWM and CHIP
    # overlaps[1] = Overlap between PWM and FOOT
    for i in range(0,len(overlaps[0])):
        if(overlaps[0][i] and overlaps[1][i]): top_1T_1F.append([chrom] + pwmList[i])
        elif(overlaps[0][i] and (not overlaps[1][i])): top_1T_0F.append([chrom] + pwmList[i])
        elif((not overlaps[0][i]) and overlaps[1][i]): top_0T_1F.append([chrom] + pwmList[i])
        elif((not overlaps[0][i]) and (not overlaps[1][i])): top_0T_0F.append([chrom] + pwmList[i])

# End for chrom in chromSet

# Sorting result lists by the motif matching score

top_1T_1F = sorted(top_1T_1F,key = lambda k: k[4])
top_1T_0F = sorted(top_1T_0F,key = lambda k: k[4])
top_0T_1F = sorted(top_0T_1F,key = lambda k: k[4])
top_0T_0F = sorted(top_0T_0F,key = lambda k: k[4])


#########################################
##### Writing Results
#########################################

# File handling
outputFile_1T_1F = open(outputLocation+"top"+str(min(len(top_1T_1F),topN))+"_1T_1F.bed","w")
outputFile_1T_0F = open(outputLocation+"top"+str(min(len(top_1T_0F),topN))+"_1T_0F.bed","w")
outputFile_0T_1F = open(outputLocation+"top"+str(min(len(top_0T_1F),topN))+"_0T_1F.bed","w")
outputFile_0T_0F = open(outputLocation+"top"+str(min(len(top_0T_0F),topN))+"_0T_0F.bed","w")

# Writing results
topList = [top_1T_1F, top_1T_0F, top_0T_1F, top_0T_0F]
fileList = [outputFile_1T_1F, outputFile_1T_0F, outputFile_0T_1F, outputFile_0T_0F]
for i in range(0,len(topList)):
    for j in range(len(topList[i])-1,max(len(topList[i])-topN-1,-1),-1):
        fileList[i].write(topList[i][j][0]+" "+str(topList[i][j][1])+" "+str(topList[i][j][2])+" "+topList[i][j][3]+" "+str(topList[i][j][4])+" "+topList[i][j][5]+"\n")

# Termination
outputFile_1T_1F.close()
outputFile_1T_0F.close()
outputFile_0T_1F.close()
outputFile_0T_0F.close()



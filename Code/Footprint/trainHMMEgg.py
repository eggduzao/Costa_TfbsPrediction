#################################################################################################
# Trains a multivariate HMM based on an annotated sequence. The training is made based on maximum-
# likelihood estimation. It uses an stt file to train the model. 
# The stt file follows this format: <chrName> <pos1> <pos2> <AnnSeq>
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. clearNonExistingTrans = If y then put 0 in all the transitions that were")
params.append("                               0 in the starterModel. If n ignores this idea.")
params.append("    2. dataModeList = Mode of training data. n = normal, l = log.")
params.append("    3. starterModel = Initial hmm models (pseudocounts).")
params.append("    4. trainSet = An stt file containing the training sequences.")
params.append("    5. signalList = Location of the signal files separated by comma.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. model.hmm = Resulting models.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import math
import copy
import glob
import operator
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
clearNonExistingTrans = sys.argv[1]
dataModeList = sys.argv[2]
starterModel = sys.argv[3]
trainSet = sys.argv[4]
signalList = sys.argv[5]
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation += "/"

# Creating signalList and dataMode list
dataModeList = dataModeList.split(",")
signalList = signalList.split(",")

# Creating starterModel
stateNo, dimNo, piList, transList, emissList = hmmFunctions.createHMM(starterModel,returnMode="mat")

# Storing HS positions into dictionary
trainDict = aux.createBedDictFromSingleFile(trainSet, features=[1,2,3], vectorize=[3])

# Sorting dictionary vectors by pos1 to allow quick access
trainDict = gsort.sortBedDictionaries([trainDict])[0]

# Creating chrNameList
chrNameList = constants.getChromList(reference=[trainDict])

# Initializing minimal descriptors
stateCountList = [ [0.0]*dimNo for k in range(0,stateNo)]
xSumMats = [ [0.0]*dimNo for k in range(0,stateNo)]
xMulMats = [ [ [0.0]*dimNo for i in range(0,dimNo)] for k in range(0,stateNo)]

# Iterating on the training set positions
for chrName in chrNameList:

    # Obtaining bw object
    inputFiles = []
    bwList = []
    for i in range(0,len(signalList)):
        inputFiles.append(open(signalList[i],"r"))
        bwList.append(BigWigFile(inputFiles[-1]))

    # Iterating on coordDict
    for coord in trainDict[chrName]:

        annSeq = coord[2]

        # Fetching sequences
        sequences = []
        for i in range(0,len(signalList)):
            bwQuery = aux.correctBW(bwList[i].get(chrName,coord[0],coord[1]),coord[0],coord[1])
            if(bwQuery == None or bwQuery == []): continue
            sequence = []
            for value in bwQuery:
                if(dataModeList[i] == "l"): value = math.log(value+1)
                sequence.append(value)
            sequences.append(sequence)

        # Counting model parameters
        for i in range(0,len(sequences)):

            piList[annSeq[0]] += 1.0
            stateCountList[annSeq[0]][i] += 1.0

            xSumMats[annSeq[0]][i] += sequences[i][0]
            for j in range(0,len(sequences)): xMulMats[annSeq[0]][i][j] += (sequences[i][0] * sequences[j][0])

            for k in range(1,len(annSeq)):

                stateCountList[annSeq[k]][i] += 1.0
                xSumMats[annSeq[k]][i] += sequences[i][k]
                
                for j in range(0,len(sequences)): xMulMats[annSeq[k]][i][j] += (sequences[i][k] * sequences[j][k])

                if(clearNonExistingTrans == "n" or transList[annSeq[k-1]][annSeq[k]] > 0.0): 
                    transList[annSeq[k-1]][annSeq[k]] += 1.0
        # End for i in range(0,len(sequences))

    # Closing signal files
    for f in inputFiles: f.close()

# End for chrName in chrNameList

# Evaluating model parameters

# Initial Probabilities
piSum = sum(piList)
for i in range(0,len(piList)): 
    if(piSum > 0.0): piList[i] = round(piList[i] / piSum, 4)

# Emissions
for k in range(0,len(xSumMats)):

    mean = [0.0] * len(xSumMats[k])
    covMat = [ [0.001]*dimNo for i in range(0,dimNo)]
    for i in range(0,len(xSumMats[k])):
        if(stateCountList[k][i] > 0.0): mean[i] = round(xSumMats[k][i] / stateCountList[k][i], 4)
        for j in range(0,len(xSumMats[k])):
            if(stateCountList[k][i] > 0.0): covMat[i][j] = round( (1/(stateCountList[k][i]-1)) * ( xMulMats[k][i][j] - ((xSumMats[k][i]*xSumMats[k][j])/stateCountList[k][i]) ) ,4)

    emissList[k][0] = mean
    counter = 0
    for i in range(0,len(xSumMats[k])):
        for j in range(0,len(xSumMats[k])):
            emissList[k][1][counter] = covMat[i][j]
            counter += 1

# Transitions
for i in range(0,len(transList)):
    transSum = sum(transList[i])
    if(transSum > 0.0):
        total = 0.0 # This total summ of round numbers will serve to make that the sum of this line converge to 1
        for j in range(0,len(transList[i])):
            total += round(transList[i][j] / transSum, 4)
            transList[i][j] = round(transList[i][j] / transSum, 4)
        if(total != 1.0): transList[i][i] += (1.0 - total) # Adjusting summation to 1 using the self transition

# Writing the model to the .hmm file
hmmFunctions.writeHMM(piList, transList, emissList, outputLocation+"model.hmm")



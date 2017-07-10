##################################################
## Statistics on TFBS peaks and motifs
##################################################

# Import
import os
import sys
import aux
import glob
import Table
import numpy as np
from Bio import Motif

# Input & output
cellLineList = sys.argv[1].split(",") # Eg. H1hesc,K562
fdrList = [float(e) for e in sys.argv[2].split(",")] # Eg. 0.001,0.0001
evNameList = sys.argv[3].split(",") # Eg. fdr_3,fdr_4
pwmLocation = sys.argv[4] # Eg. "/hpcwork/izkf/projects/egg/Data/PWM/"
tfbsLocation = sys.argv[5] # "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBS/"
mpbsLocation = sys.argv[6] # "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBS/"
outputFileName = sys.argv[7] # Name of the output (a tex file)
totalExt = int(sys.argv[8])

# Additional parameters
pwmPseudocounts = 0.00
pwmPrecision = 10**4
outputLocation = "/".join(outputFileName.split("/")[:-1])+"/"
headerVec = ["Factor","Bitscore","MPBS","ChIP","MPBS+ChIP","ChIP+MPBS"]

# Iterating on cell lines
dataMatrixList = []
captionList = []
for cellLine in cellLineList:

    # ChIP TFBS location
    tfbsLoc = tfbsLocation+cellLine+"/"
    factorList = [e.split("/")[-1].split(".")[0] for e in glob.glob(tfbsLoc+"*.bed")]
    factorList.sort()

    # Iterating on evidence names list
    for i in range(0,len(evNameList)):

        # MPBS location
        fdr = fdrList[i]
        evName = evNameList[i]
        mpbsLoc = mpbsLocation+cellLine+"_Evidence/"+evName+"/"

        # Iterating on factors
        dataMatrix = []
        captionList.append(" ".join([cellLine]+evName.split("_")))
        for factor in factorList:

            # Initializing
            dataVec = [factor]
            mpbsFileName = mpbsLoc+factor+".bed"
            tfbsFileName = tfbsLoc+factor+".bed"

            # Verifying if all files exist
            if(not os.path.isfile(mpbsFileName) or not os.path.isfile(tfbsFileName)):
                dataVec[0] += "\\**"
                for e in range(0,5): dataVec.append("--")
                continue

            # Calculating bitscore
            dataVec.append(aux.calculateBitscore(pwmLocation,cellLine,factor,pwmPseudocounts,pwmPrecision,fdr,outputLocation))
    
            # Calculating number of MPBS and ChIP
            totalMPBS = aux.fileLen(mpbsFileName)
            totalTFBS = aux.fileLen(tfbsFileName)
            dataVec.append(totalMPBS)
            dataVec.append(totalTFBS)

            # Calculating MPBS + ChIP
            mpbsFile = open(mpbsFileName)
            posCount = 0
            for line in mpbsFile:
                ll = line.strip().split("\t")
                if(ll[3].split(":")[1] == "Y"): posCount += 1
            mpbsFile.close()
            dataVec.append(str(posCount)+" ("+str(round((100*float(posCount))/float(totalMPBS),2))+")")

            # Calculating ChIP-mpbs intersections
            chipMpbsCount = aux.fileLenIntersect(tfbsFileName,mpbsFileName,False,outputLocation+cellLine+evName+factor+"_itemp1.bed",
                            outputLocation+cellLine+evName+factor+"_itemp2.bed",totalExt)
            dataVec.append(str(chipMpbsCount)+" ("+str(round((100*float(chipMpbsCount))/float(totalTFBS),2))+")")
            
            # Updating dataMatrix
            dataMatrix.append(dataVec)

        # Updating dataMatrixList
        dataMatrix = [list(e) for e in np.array(dataMatrix).T]
        dataMatrixList.append(dataMatrix)

# Creating table
aux.createTable(captionList,headerVec,dataMatrixList,outputFileName)



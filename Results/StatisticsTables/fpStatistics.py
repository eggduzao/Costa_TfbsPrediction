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
cellLine = sys.argv[1] # Eg. H1hesc
evName = sys.argv[2] # Eg. fdr_4
tfbsLocation = sys.argv[3] # "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/TFBSAWG/"
footLocation = sys.argv[4] # "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Results/"
outputFileName = sys.argv[5] # Name of the output (a tex file)

# Additional parameters
tfbsLoc = tfbsLocation+cellLine+"/"
footLoc = footLocation+cellLine+"/"+evName+"/"
outputLocation = "/".join(outputFileName.split("/")[:-1])+"/"
#headerVec = ["Factor","FP+ChIP","FP-ChIP","ChIP+FP","ChIP-FP","TP","FP","TN","FN","Acc","BAc","Sens","Spec","PPV","NPV"]
headerVec = ["Factor","TP","FP","TN","FN","Acc","BAc","Sens","Spec","PPV","NPV"]
factorList = [e.split("/")[-1].split(".")[0] for e in glob.glob(tfbsLoc+"*.bed")]
factorList.sort()

# Iterating on result files
dataMatrixList = []
captionList = []
for footFileName in glob.glob(footLoc+"*.bed"):

    # Threshold loop
    threshList = [1001.0]
    if("Pique" in footFileName): threshList = [0.95,0.99]
    for thresh in threshList:

        # Initializing
        dataMatrix = []
        captionList.append(" ".join(footFileName.split("/")[-1].split(".")[0].split("_"))+" "+str(thresh))

        # Iterating on factors
        for factor in factorList:

            # Initializing
            dataVec = [factor]
            tfbsFileName = tfbsLoc+factor+".bed"

            # Calculating intersection FP / ChIP
            """
            tempFileName = outputLocation+cellLine+"_"+evName+"_"+factor+"_fptemp.bed"
            tempFile = open(tempFileName,"w")
            footFile = open(footFileName,"r")
            for line in footFile:
                score = float(line.strip().split("\t")[4])
                if(score > thresh): tempFile.write(line)
            footFile.close()
            tempFile.close()
            dataVec.append(aux.fileLenIntersect(tempFileName,tfbsFileName,False,outputLocation+cellLine+evName+factor+"_itemp1.bed"))
            dataVec.append(aux.fileLenIntersect(tempFileName,tfbsFileName,True,outputLocation+cellLine+evName+factor+"_itemp2.bed"))
            dataVec.append(aux.fileLenIntersect(tfbsFileName,tempFileName,False,outputLocation+cellLine+evName+factor+"_itemp3.bed"))
            dataVec.append(aux.fileLenIntersect(tfbsFileName,tempFileName,True,outputLocation+cellLine+evName+factor+"_itemp4.bed"))
            os.system("rm "+tempFileName)
            """

            # Calculating contingency table and statistics
            footFile = open(footFileName,"r")
            tp = 0.0; fp = 0.0; tn = 0.0; fn = 0.0
            for line in footFile:
                currFacName = line.strip().split("\t")[3].split(":")[0]
                if(factor != currFacName): continue
                score = float(line.strip().split("\t")[4])
                ev = line.strip().split("\t")[3].split(":")[1]
                if(ev == "Y" and score >= thresh): tp += 1.0
                elif(ev == "Y" and score < thresh): fn += 1.0
                elif(ev == "N" and score >= thresh): fp += 1.0
                elif(ev == "N" and score < thresh): tn += 1.0
                elif(ev == "." and score >= thresh): fp += 1.0
                elif(ev == "." and score < thresh): tn += 1.0
            footFile.close()
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

            # Updating dataMatrix
            dataMatrix.append(dataVec)

        # Updating dataMatrixList
        dataMatrix = [list(e) for e in np.array(dataMatrix).T]
        dataMatrixList.append(dataMatrix)

# Creating table
aux.createTable(captionList,headerVec,dataMatrixList,outputFileName)



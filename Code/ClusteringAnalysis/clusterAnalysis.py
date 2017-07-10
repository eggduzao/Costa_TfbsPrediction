#################################################################################################
# Performs clustering of vectors.
#### Algorithms:
# k = K-means Algorithm   |   h = Hierarchical Algorithm
#### Methods:
# On k-means: mean = k-means   |   median = K-medians
# On hierarchical: sl = Single Linkage  |  cl = Complete Linkage  |  el = Centroid Linkage  |  al = Average Linkage
#### Distance:
# corr = Correlation  |  abscorr = Absolute correlation  |  uncentcorr = Uncentered correlation
# absunccorr = Absolute uncentered correlation  |  spearman = Spearman  |  kendall = Kendall
# euc = Euclidean  |  cityblock = City-block
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. algorithm = The clustering algorithm.")
params.append("    2. method = The method based on the algorithm.")
params.append("    3. distance = The distance used on the algorithms.")
params.append("    4. noClust = Number of clusters.")
params.append("    5. inputFileName = File (post) containing the numerical heatmap.")
params.append("    6. outputLocationStat = Location of the output and temporary files for stat.")
params.append("    7. outputLocationPost = Location of the output and temporary files for post.")
params.append("    8. outputLocationHeat = Location of the output and temporary files for heat.")
params.append("###")
params.append("Output: ")
params.append("    1. <inp>_<alg>_<met>_<dist>_<noClust>.stat = Results (statistics).")
params.append("    2. <inp>_<clType>_<alg>_<met>_<dist>_<noClust>.post = Clustered post file.")
params.append("    3. <inp>_<clType>_<alg>_<met>_<dist>_<noClust>.heat = Clustered heat file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import copy
import operator
import numpy as np
import Pycluster as pc
from sklearn import metrics
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
algorithm = sys.argv[1]
method = sys.argv[2]
distance = sys.argv[3]
noClust = int(sys.argv[4])
inputFileName = sys.argv[5]
outputLocationStat = sys.argv[6]
if(outputLocationStat[-1] != "/"): outputLocationStat+="/"
outputLocationPost = sys.argv[7]
if(outputLocationPost[-1] != "/"): outputLocationPost+="/"
outputLocationHeat = sys.argv[8]
if(outputLocationHeat[-1] != "/"): outputLocationHeat+="/"

##################################################
### Reading data
##################################################

# File handling
inputFile = open(inputFileName,"r")

# Evidence list
evidenceList = list(inputFile.readline().strip())

# Main data structures
rawData = []
for i in range(0,len(evidenceList)): rawData.append([])
labelList = []
counter = -1

# Data reading loop
for line in inputFile:
    if(line[0] == "#"):
        labelList.append(line[1:].strip())
        counter = 0
        continue
    ll = [float(e) for e in line.strip().split(" ")]
    for e in ll: rawData[counter].append(e)
    counter += 1
inputFile.close()

# Active list numbers
activeList = [labelList.index(e) for e in ["DNase","H2AZ","H3K27ac","H3K9ac","H3K4me2","H3K4me3"]]

# Determining fragLen (the length in bp of the fragment)
rawVph = (len(rawData[0]) / len(labelList)) # Values per histone
fragLen = rawVph / 5

##################################################
### Clustering data
##################################################

# Distance metrics
dDict = dict([("corr","c"),("abscorr","a"),("uncentcorr","u"),("absunccorr","x"),("spearman","s"),("kendall","k"),("euc","e"),("cityblock","b")])

# Unsupervised validation metrics list
silhouetteList = []

# K-means
if(algorithm == "k"):
    # Method
    mDict = dict([("mean","a"),("median","m")])
    # All
    clusterListAll, error, nfound = pc.kcluster(np.array(rawData), nclusters=noClust, transpose=0, method=mDict[method], dist=dDict[distance])
    silScore = metrics.silhouette_score(rawData, clusterListAll, metric='euclidean')
    silhouetteList.append(silScore)
    # Single
    clusterListSingle = []
    for i in range(0,len(labelList)):
        cluterListTemp, error, nfound = pc.kcluster(np.array(rawData)[:,(rawVph*i):(rawVph*i)+rawVph], nclusters=noClust, transpose=0, method=mDict[method], dist=dDict[distance])
        clusterListSingle.append(cluterListTemp)
        silScore = metrics.silhouette_score(np.array(rawData)[:,(rawVph*i):(rawVph*i)+rawVph], cluterListTemp, metric='euclidean')
        silhouetteList.append(silScore)
    # Active Histones
    actHistList = []
    for i in activeList[1:]:
        for j in range(0,rawVph): actHistList.append((rawVph*i)+j);
    clusterListActiveHist, error, nfound = pc.kcluster(np.array(rawData)[:,actHistList], nclusters=noClust, transpose=0, method=mDict[method], dist=dDict[distance])
    silScore = metrics.silhouette_score(np.array(rawData)[:,actHistList], clusterListActiveHist, metric='euclidean')
    silhouetteList.append(silScore)
    # Active Histones + DNase
    actHistDNaseList = []
    for i in activeList:
        for j in range(0,rawVph): actHistDNaseList.append((rawVph*i)+j);
    clusterListActiveHistDNase, error, nfound = pc.kcluster(np.array(rawData)[:,actHistDNaseList], nclusters=noClust, transpose=0, method=mDict[method], dist=dDict[distance])
    silScore = metrics.silhouette_score(np.array(rawData)[:,actHistDNaseList], clusterListActiveHistDNase, metric='euclidean')
    silhouetteList.append(silScore)

# Hierarchical
if(algorithm == "h"):
    # Method
    mDict = dict([("sl","s"),("cl","m"),("el","c"),("al","a")])
    # All
    tree = pc.treecluster(np.array(rawData), transpose=0, method=mDict[method], dist=dDict[distance])
    clusterListAll = tree.cut(noClust)
    silScore = metrics.silhouette_score(rawData, clusterListAll, metric='euclidean')
    silhouetteList.append(silScore)
    # Single
    clusterListSingle = []
    for i in range(0,len(labelList)):
        tree = pc.treecluster(np.array(rawData)[:,(rawVph*i):(rawVph*i)+rawVph], transpose=0, method=mDict[method], dist=dDict[distance])
        cluterListTemp = tree.cut(noClust)
        clusterListSingle.append(cluterListTemp)
        silScore = metrics.silhouette_score(np.array(rawData)[:,(rawVph*i):(rawVph*i)+rawVph], cluterListTemp, metric='euclidean')
        silhouetteList.append(silScore)
    # Active Histones
    actHistList = []
    for i in activeList[1:]:
        for j in range(0,rawVph): actHistList.append((rawVph*i)+j);
    tree = pc.treecluster(np.array(rawData)[:,actHistList], transpose=0, method=mDict[method], dist=dDict[distance])
    clusterListActiveHist = tree.cut(noClust)
    silScore = metrics.silhouette_score(np.array(rawData)[:,actHistList], clusterListActiveHist, metric='euclidean')
    silhouetteList.append(silScore)
    # Active Histones + DNase
    actHistDNaseList = []
    for i in activeList:
        for j in range(0,rawVph): actHistDNaseList.append((rawVph*i)+j);
    tree = pc.treecluster(np.array(rawData)[:,actHistDNaseList], transpose=0, method=mDict[method], dist=dDict[distance])
    clusterListActiveHistDNase = tree.cut(noClust)
    silScore = metrics.silhouette_score(np.array(rawData)[:,actHistDNaseList], clusterListActiveHistDNase, metric='euclidean')
    silhouetteList.append(silScore)

##################################################
### Statistics
# ORDER = [TP, FP, TN, FN]
##################################################

# All results:
allClusters = [clusterListAll] + clusterListSingle + [clusterListActiveHist] + [clusterListActiveHistDNase]
allClusters = np.array(allClusters)
allLabels = ["All"] + labelList + ["Active Histones"] + ["Active Histones + DNase"]
allResults = []
for i in range(0,len(allClusters)): allResults.append([0.0,0.0,0.0,0.0])
evidenceList = np.array(evidenceList)

# Counting confusion matrix
highClusterNo = []; lowClusterNo = []
for i in range(0,len(allClusters)):
    sum0Y = float(np.sum(evidenceList[allClusters[i] == 0] == "Y"))
    sum0N = float(np.sum(evidenceList[allClusters[i] == 0] == "N"))
    sum1Y = float(np.sum(evidenceList[allClusters[i] == 1] == "Y"))
    sum1N = float(np.sum(evidenceList[allClusters[i] == 1] == "N"))
    if(sum0Y/(sum0Y+sum0N) >= sum1Y/(sum1Y+sum1N)):
        highClusterNo.append(0); lowClusterNo.append(1)
    else:
        highClusterNo.append(1); lowClusterNo.append(0)
    allResults[i][0] = float(np.sum(evidenceList[allClusters[i] == highClusterNo[i]] == "Y")) # TP
    allResults[i][1] = float(np.sum(evidenceList[allClusters[i] == highClusterNo[i]] == "N")) # FP
    allResults[i][2] = float(np.sum(evidenceList[allClusters[i] == lowClusterNo[i]] == "N")) # TN
    allResults[i][3] = float(np.sum(evidenceList[allClusters[i] == lowClusterNo[i]] == "Y")) # FN

# Evaluating cluster validation scores
# Each valList element contains [Adjusted Rand, Mutual Info, Adjusted Mutual Info, Normalized Mutual Info, Homogeneity, Completeness, V Measure]
valList = []
for i in range(0,len(allClusters)):
    valItem = []
    valItem.append(metrics.adjusted_rand_score(evidenceList, allClusters[i]))
    valItem.append(metrics.mutual_info_score(evidenceList, allClusters[i]))
    valItem.append(metrics.adjusted_mutual_info_score(evidenceList, allClusters[i]))
    valItem.append(metrics.normalized_mutual_info_score(evidenceList, allClusters[i]))
    valItem.append(metrics.homogeneity_score(evidenceList, allClusters[i]))
    valItem.append(metrics.completeness_score(evidenceList, allClusters[i]))
    valItem.append(metrics.v_measure_score(evidenceList, allClusters[i]))
    valList.append(valItem)

# Writing results
statFile = open(outputLocationStat+inputFileName.split("/")[-1].split(".")[0]+"_"+algorithm+"_"+method+"_"+distance+"_"+str(noClust)+".stat","w")
for i in range(0,len(allResults)):
    statFile.write("# "+allLabels[i]+"\n")
    tp = allResults[i][0]; fp = allResults[i][1]; tn = allResults[i][2]; fn = allResults[i][3]
    statFile.write("TP = "+str(tp)+"\t"+"FP = "+str(fp)+"\t"+"TN = "+str(tn)+"\t"+"FN = "+str(fn)+"\n")
    if((tp+tn+fp+fn) > 0): statFile.write("CorrectRate = "+str((tp+tn)/(tp+tn+fp+fn))+"\n")
    else: statFile.write("CorrectRate = -----\n")
    if((tp+fn) > 0): statFile.write("Sensitivity = "+str(tp/(tp+fn))+"\n")
    else: statFile.write("Sensitivity = -----\n")
    if((tn+fp) > 0): statFile.write("Specificity = "+str(tn/(tn+fp))+"\n")
    else: statFile.write("Specificity = -----\n")
    if((tp+fp) > 0): statFile.write("PPV = "+str(tp/(tp+fp))+"\n")
    else: statFile.write("PPV = -----\n")
    if((tn+fn) > 0): statFile.write("NPV = "+str(tn/(tn+fn))+"\n")
    else: statFile.write("NPV = -----\n")
    statFile.write("Silhouette = "+str(silhouetteList[i])+"\n")
    statFile.write("Adjusted Rand = "+str(valList[i][0])+"\n")
    statFile.write("Mutual Info = "+str(valList[i][1])+"\n")
    statFile.write("Adjusted Mutual Info = "+str(valList[i][2])+"\n")
    statFile.write("Normalized Mutual Info = "+str(valList[i][3])+"\n")
    statFile.write("Homogeneity = "+str(valList[i][4])+"\n")
    statFile.write("Completeness = "+str(valList[i][5])+"\n")
    statFile.write("V Measure = "+str(valList[i][6])+"\n\n")
statFile.close()

##################################################
### Writing data
##################################################

# File handling
allLabelsReduced = ["All"] + labelList + ["ActHist"] + ["ActHistDNase"]
postList = [open(outputLocationPost+inputFileName.split("/")[-1].split(".")[0]+"_"+e+"_"+algorithm+"_"+method+"_"+distance+"_"+str(noClust)+".post","w") for e in allLabelsReduced]
heatList = [open(outputLocationHeat+inputFileName.split("/")[-1].split(".")[0]+"_"+e+"_"+algorithm+"_"+method+"_"+distance+"_"+str(noClust)+".heat","w") for e in allLabelsReduced]

# Making numeric evidenceList
for i in range(0,len(evidenceList)):
    if(evidenceList[i] == "N"): evidenceList[i] = 0.0
    else: evidenceList[i] = 1.0
evidenceList = [float(e) for e in evidenceList]

# Red/Green vectors:
greenVec = "20.0 " * fragLen; greenVec = greenVec[:-1]
redVec = "40.0 " * fragLen; redVec = redVec[:-1]

# Color dictionary
# BACK (black):  0.2 - 10.0  ->  Estados: 0
# UP (green):   10.2 - 20.0  ->  Estados: 1
# TOP (blue):   20.2 - 30.0  ->  Estados: 2
# DOWN (red):   30.2 - 40.0  ->  Estados: 3
# DIP (yellow): 40.2 - 50.0  ->  Estados: 4
minDict = dict([(0,0.2), (1,10.2), (2,20.2), (3,30.2), (4,40.2)])
maxDict = dict([(0,10.0), (1,20.0), (2,30.0), (3,40.0), (4,50.0)])

# Iterating on various cluster options
for fCount in range(0,len(postList)):

    postFile = postList[fCount]
    heatFile = heatList[fCount]

    # Sorting rawData
    rawTemp = np.append(np.array([[e] for e in allClusters[fCount]]),copy.deepcopy(rawData),1)
    rawTemp = np.append(np.array([[e] for e in evidenceList]),rawTemp,1).tolist()
    rawTemp.sort(key=lambda x: x[1])

    # Writing post ########

    # Evidence
    for i in range(0,len(rawTemp)):
        if(rawTemp[i][0] == 0): postFile.write("N")
        else: postFile.write("Y")
    postFile.write("\n")

    # Data
    for j in range(0,len(labelList)):
        postFile.write("#"+labelList[j]+"\n")
        for i in range(0,len(evidenceList)):
            fragVec = rawTemp[i][(j*rawVph)+2:(j*rawVph)+2+rawVph]
            postFile.write(str(fragVec[0]))
            for e in fragVec[1:]: postFile.write(" "+str(e))
            postFile.write("\n")    

    # Writing heat ########

    # Labels
    heatFile.write(labelList[0])
    for e in labelList[1:]:
        heatFile.write(" "+e)
    heatFile.write("\n")

    # Evidence
    if(rawTemp[0][0] == 0): heatFile.write(redVec)
    else: heatFile.write(greenVec)
    for i in range(1,len(rawTemp)):
        if(rawTemp[i][0] == 0): heatFile.write(" "+redVec)
        else: heatFile.write(" "+greenVec)
    heatFile.write("\n")

    # Clusters
    if(rawTemp[0][1] == lowClusterNo[fCount]): heatFile.write(redVec)
    else: heatFile.write(greenVec)
    for i in range(1,len(rawTemp)):
        if(rawTemp[i][1] == lowClusterNo[fCount]): heatFile.write(" "+redVec)
        else: heatFile.write(" "+greenVec)
    heatFile.write("\n")

    # Data
    for j in range(0,len(labelList)):
        currLabel = rawTemp[0][0]
        for i in range(0,len(evidenceList)):
            fragVec = rawTemp[i][(j*rawVph)+2:(j*rawVph)+2+rawVph]
            toWrite = []
            for k in range(0,len(fragVec),len(minDict)):
                tempFrag = fragVec[k:k+len(minDict)]
                s = tempFrag.index(max(tempFrag))
                toWrite.append( round( ( max(tempFrag) * (maxDict[s]-minDict[s]) ) + minDict[s],4) )
            if(i == 0):
                heatFile.write(str(toWrite[0]))
                for e in toWrite[1:]: heatFile.write(" "+str(e))
            else:
                for e in toWrite: heatFile.write(" "+str(e))
        heatFile.write("\n")

# Termination
for e in postList: e.close()
for e in heatList: e.close()



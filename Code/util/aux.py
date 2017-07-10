#################################################################################################
# Contains miscelaneous auxiliary functions.
#################################################################################################

# Import
import constants
import os

def isInteger(s):
    """ 
    Returns True if string s is an integer and False if it is not.
      s: Any string.
    """
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False
# End isInteger

def isFloat(s):
    """ 
    Returns True if string s is a float and False if it is not.
      s: Any string.
    """
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
# End isFloat

def vectorizeByCol(mat):
    """ 
    Creates a vector representation of a matrix by column (entry for multivariate HMMs)
      mat: Matrix (vector of vectors).
    """
    vec = []
    for j in range(0,len(mat[0])):
        for i in range(0,len(mat)):
            vec.append(mat[i][j])
    return vec
# End vectorizeByCol

def checkTuplesOverlap(t1, t2):
    """ 
    Checks if one interval contains any overlap with another interval.
    Returns -1 if i1 is before i2; 1 if i1 is after i2; and 0 if they contain any overlap.
      t1: First tuple.
      t2: Second tuple.
    """
    if(t1[1] <= t2[0]): return -1 # interval1 is before interval2
    if(t2[1] <= t1[0]): return 1 # interval1 is after interval2
    return 0 # interval1 overlaps interval2
# End checkTuplesOverlap

def mergeBedEntries(p1, p2):
    """ 
    Merges bed or pk entries when it is already known that they overlap.
    The chromossome field (first field) must not be present. The fields pos1, pos2, name, score and strand must be present.
      p1: First entry.
      p2: Second entry.
    """
    np = []
    np.append(min(p1[0],p2[0])); np.append(max(p1[1],p2[1])); np.append(p1[2]); np.append(max(p1[3],p2[3])); np.append(p1[4])
    if(len(p1) > 5): np.append(max(p1[5],p2[5])) 
    if(len(p1) > 6): np.append(max(p1[6],p2[6])) 
    if(len(p1) > 7): np.append(max(p1[7],p2[7]))
    if(len(p1) > 8): np.append(max(p1[8],p2[8]))
    return np
# End mergeBedEntries

def createBedDictFromSingleFile(coordFileName, features=[], vectorize=[], separator=" "):
    """
    Creates a dictionary from a coordinate (bed or pk) which keys are the chromossomes and the elements are lists of selected bed features.
      coordFileName: Location + name of the bed or pk file.
      features: List containing the features positions to store. If empty (default), stores all features.
      vectorize: List containing the features to be vectorized.
      separator: Character used to separate the features in the bed or pk file. Default is empty space.
    """
    coordFile = open(coordFileName,"r")
    coordDict = dict()
    for line in coordFile:
        if(len(line) < 2): continue
        lineList = line.strip().split(separator)
        chrName = lineList[0]
        if(not features): feats = lineList[1:]
        else: feats = [lineList[i] for i in features]
        for i in range(0,len(feats)):
            if((i+1) in vectorize and isInteger(feats[i][0])): feats[i] = [int(e) for e in feats[i]]
            elif((i+1) in vectorize): feats[i] = [e for e in feats[i]]
            elif(isInteger(feats[i])): feats[i] = int(feats[i])
            elif(isFloat(feats[i])): feats[i] = float(feats[i])
        if(chrName not in coordDict.keys()): coordDict[chrName] = [feats]
        else: coordDict[chrName].append(feats)      
    coordFile.close()
    return coordDict
# End createBedDictFromSingleFile

def createBedDictFromMultipleFile(coordFileLocation, features=[], separator=" "):
    """ 
    Creates a dictionary from multiple coordinate (bed or pk) files, which keys are the chromossomes and the elements are lists of selected bed features.
    Each file must be named with its respective chromossome name.
      coordFileLocation: Location of the bed or pk files (by chromossome).
      features: List containing the features positions to store. If empty (default), stores all features.
      separator: Character used to separate the features in the bed or pk file. Default is empty space.
    """
    chromList = constants.getChromList()
    coordDict = dict()
    for chrom in chromList:
        if(os.path.isfile(coordFileLocation+chrom+".bed")): coordFile = open(coordFileLocation+chrom+".bed","r")
        elif(os.path.isfile(coordFileLocation+chrom+".pk")): coordFile = open(coordFileLocation+chrom+".pk","r")
        else: continue
        for line in coordFile:
            if(len(line) < 2): continue
            lineList = line.strip().split(separator)
            chrName = lineList[0]
            if(not features): feats = lineList[1:]
            else: feats = [lineList[i] for i in features]
            for i in range(0,len(feats)):
                if(isInteger(feats[i])): feats[i] = int(feats[i])
                elif(isFloat(feats[i])): feats[i] = float(feats[i])
            if(chrName not in coordDict.keys()): coordDict[chrName] = [feats]
            else: coordDict[chrName].append(feats)
        coordFile.close()
    return coordDict
# End createBedDictFromMultipleFile

def createBedListFromSingleFile(coordFileName, features=[], separator=" "):
    """
    Creates a list of list of bed features according to the given input file.
      coordFileName: Location + name of the bed or pk file.
      features: List containing the features positions to store. If empty (default), stores all features.
      separator: Character used to separate the features in the bed or pk file. Default is empty space.
    """
    coordFile = open(coordFileName,"r")
    coordList = []
    for line in coordFile:
        if(len(line) < 2): continue
        lineList = line.strip().split(separator)
        if(not features): feats = lineList[0:]
        else: feats = [lineList[i] for i in features]
        for i in range(0,len(feats)):
            if(isInteger(feats[i])): feats[i] = int(feats[i])
            elif(isFloat(feats[i])): feats[i] = float(feats[i])
        coordList.append(feats)
    coordFile.close()
    return coordList    
# End createBedListFromSingleFile

def createBedListFromMultipleFile(coordFileLocation, features=[], separator=" "):
    """ 
    Creates a list from multiple coordinate (bed or pk) files.
      coordFileLocation: Location of the bed or pk files (by chromossome).
      features: List containing the features positions to store. If empty (default), stores all features.
      separator: Character used to separate the features in the bed or pk file. Default is empty space.
    """
    chromList = constants.getChromList()
    coordList = []
    for chrom in chromList:
        if(os.path.isfile(coordFileLocation+chrom+".bed")): coordFile = open(coordFileLocation+chrom+".bed","r")
        elif(os.path.isfile(coordFileLocation+chrom+".pk")): coordFile = open(coordFileLocation+chrom+".pk","r")
        else: continue
        for line in coordFile:
            if(len(line) < 2): continue
            lineList = line.strip().split(separator)
            chrName = lineList[0]
            if(not features): feats = lineList[1:]
            else: feats = [lineList[i] for i in features]
            for i in range(0,len(feats)):
                if(isInteger(feats[i])): feats[i] = int(feats[i])
                elif(isFloat(feats[i])): feats[i] = float(feats[i])
            coordList.append([chrName]+feats)
        coordFile.close()
    return coordList
# End createBedListFromMultipleFile

def wigToBigWig(wigFileName, bigWigFileName, removeWig=True):
    """
    Converts a wig file into a bigWig file
      wigFileName: Wig file name. Standard extension = "wig".
      bigWigFileName: BigWig file name. Standard extension = "bw".
      removeWig: If true (default) removes the wig file after conversion. If false, keep the wig file.
    """
    #os.system("wigToBigWig "+wigFileName+" "+constants.getChromSizesLocation()+" "+bigWigFileName)
    os.system("wigToBigWig "+wigFileName+" /hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes.ext10000 "+bigWigFileName)
    if(removeWig): os.system("rm -rf "+wigFileName)
    return "ok"
# End createBedListFromSingleFile

def bigWigToWig(bigWigFileName, wigFileName):
    """
    Converts a bw file into a wig file
      bigWigFileName: BigWig file name. Standard extension = "bw".
      wigFileName: Wig file name. Standard extension = "wig".
    """
    os.system("bigWigToWig "+bigWigFileName+" "+wigFileName)
    return "ok"
# End createBedListFromSingleFile

def correctBW(bwQuery,c1,c2):
    """
    Converts a bw query into a sequence where each position represent the genomic signal
    of each base pair.
      bwQuery: BigWig query.
      c1: First position of the sequence in base pairs.
      c2: Last position of the sequence (not included - according to bed format) in base pairs.
    """
    bwRes = [0.0 for e in range(c1,c2)]
    if(bwQuery):
        for (p1,p2,v) in bwQuery:
            for i in range(p1,p2): bwRes[i-c1] = v
    return bwRes

def diagonalize(matrixList):
    """
    Return a list with vectors corresponding to the diagonal of each matrix.
      matrixList: List of matrices.
    """
    resList = []
    for m in matrixList: resList.append([m[i][i] for i in range(0,len(m))])
    return resList



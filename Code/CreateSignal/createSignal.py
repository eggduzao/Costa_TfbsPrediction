#################################################################################################
# Script with the full pipeline to create the signals used by the HMM
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. slopeWindowSize = Size of the window to evaluate slope.")
params.append("    2. perNorm = Normalization percentile.")
params.append("    3. perSlope = Slope percentile.")
params.append("    4. coordFileName = Bed file with the locations to apply the signals.")
params.append("    5. signalFileName = Wig file containing the raw count signal.")
params.append("    6. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <inputFileName>_norm.bw = Resulting normalized signal.")
params.append("    2. <inputFileName>_slope.bw = Resulting slope signal.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
import math
import numpy as np
import scipy.stats as st
from bx.bbi.bigwig_file import BigWigFile
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    print lib_path
    for e in params: print e
    sys.exit(0)

#################################################################################################
# INPUT
#################################################################################################

# Reading input
slopeWindowSize = int(sys.argv[1])
perNorm = float(sys.argv[2])
perSlope = float(sys.argv[3])
coordFileName = sys.argv[4]
signalFileName = sys.argv[5]
outputLocation = sys.argv[6]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Additional input
initialClip = 1000.0 # Avoid extremes

#################################################################################################
# FUNCTIONS
#################################################################################################

# Hon (softmax) normalization
def honNorm(sequence,mean,std):
    normSeq = []
    for e in sequence:
        if(e == 0.0): normSeq.append(0.0)
        elif(e > 0.0): normSeq.append(1.0/(1.0+(np.exp(-(e-mean)/std))))
        else: normSeq.append(-1.0/(1.0+(np.exp(-(-e-mean)/std))))
    return normSeq

# Boyle normalization (modified - mean is calculated based on whole bin and not on windowing)
def boyleNorm(sequence):
    mean = np.array([e for e in sequence if e>0]).mean()
    normSeq = [(float(e)/mean) for e in sequence]
    return normSeq

# Savitzky-Golay coefficients
def savitzkyGolayCoefficients(windowSize, order, deriv):
    try:
        windowSize = np.abs(np.int(windowSize))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("windowSize and order have to be of type int")
    if windowSize % 2 != 1 or windowSize < 1:
        raise TypeError("windowSize size must be a positive odd number")
    if windowSize < order + 2:
        raise TypeError("windowSize is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (windowSize -1) // 2
    # Precompute Coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    return m[::-1]

# Evaluate slope by convolution
def slope(sequence,sgCoefs):
    slopeSeq = np.convolve(sequence,sgCoefs)
    slopeSeq = [e for e in slopeSeq[(len(sgCoefs)/2):(len(slopeSeq)-(len(sgCoefs)/2))]]
    return slopeSeq

# Evaluating statistics
def evaluateStatistics(resultingVector):
    resultingVector = np.array(resultingVector)
    mean = resultingVector.mean()
    std = resultingVector.std()
    var = resultingVector.var()
    perVec = []
    for e in percentiles: perVec.append(st.scoreatpercentile(resultingVector,e))
    return mean, std, var, perVec

# Writing results
def writeResults(mean, std, var, perVec, outputFileName):
    outputFile = open(outputFileName,"w")
    outputFile.write("Mean\t"+str(mean)+"\n")
    outputFile.write("Std\t"+str(std)+"\n")
    outputFile.write("Var\t"+str(var)+"\n")
    outputFile.write("\n".join(["\t".join(["Per"+str(percentiles[i]),str(perVec[i])]) for i in range(0,len(percentiles))]))
    outputFile.close()
    return 0

#################################################################################################
# INITIALAZING STRUCTURES
#################################################################################################

# Signal name
signalName = signalFileName.split("/")[-1].split(".")[0]

# Savitzky golay coefficients
sgCoefs = savitzkyGolayCoefficients(slopeWindowSize, 2, 1)

#################################################################################################
# SIGNAL PIPELINE
#################################################################################################

# Input files
coordFile = open(coordFileName,"r")
signalFile = open(signalFileName,"r")
bw = BigWigFile(signalFile)

# Output files
normFile = open(outputLocation+signalName+"_norm.wig","w")
slopeFile = open(outputLocation+signalName+"_slope.wig","w")

# Iterating on coordinates
for line in coordFile:

    # Reading line
    ll = line.strip().split("\t")
    chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])

    # Fetching sequence with initial clipping
    seq = np.array([min(e,initialClip) for e in aux.correctBW(bw.get(chrName,p1,p2),p1,p2)])

    # Std based clipping
    mean = seq.mean()
    std = seq.std()
    seq = [min(e,mean+(10*std)) for e in seq]

    # Boyle
    seq = np.array(boyleNorm(seq))

    # Hon - Write normalization
    perc = st.scoreatpercentile(seq,perNorm)
    std = seq.std()
    seq = honNorm(seq,perc,std)
    normFile.write("fixedStep chrom="+chrName+" start="+str(p1+1)+" step=1\n")
    for e in seq: normFile.write(str(round(e,2))+"\n")

    # Slope
    seq = slope(seq,sgCoefs)

    # Hon - Write slope
    absSeq = np.array([abs(e) for e in seq])
    perc = st.scoreatpercentile(absSeq,perSlope)
    std = absSeq.std()
    seq = honNorm(seq,perc,std)
    slopeFile.write("fixedStep chrom="+chrName+" start="+str(p1+1)+" step=1\n")
    for e in seq: slopeFile.write(str(round(e,2))+"\n")

# Closing files
coordFile.close()
signalFile.close()
normFile.close()
slopeFile.close()

# Converting to bigwig
csFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered.increased.10e6_ALL.ForBw"
os.system(" ".join(["wigToBigWig",outputLocation+signalName+"_norm.wig",csFileName,outputLocation+signalName+"_norm.bw"]))
os.system(" ".join(["wigToBigWig",outputLocation+signalName+"_slope.wig",csFileName,outputLocation+signalName+"_slope.bw"]))



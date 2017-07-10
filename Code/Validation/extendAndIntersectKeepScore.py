#################################################################################################
# Extend a prediction bed file and intersect it with a mpbs file, reporting the latter.
# However, the score of each MPBS will be associated with the prediction score.
# For this, the prediction file must have a score field.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. leftExt = Extension to the left of the prediction file.")
params.append("    2. rightExt = Extension to the right of the prediction file.")
params.append("    3. mpbsFileName = MPBS bed file.")
params.append("    4. predFileName = Prediction bed file.")
params.append("    5. outputFileName = Location and name of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Result validation bed file.")
params.append("###")
#################################################################################################

# Import
import os
import sys
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
leftExt = int(sys.argv[1])
rightExt = int(sys.argv[2])
mpbsFileName = sys.argv[3]
predFileName = sys.argv[4]
outputFileName = sys.argv[5]

# Extend prediction bed file and writes it into a new bed file
predFile = open(predFileName,"r")
bedTempFile = open(outputFileName+"_temp.bed","w")
for line in predFile:
    ll = line.strip().split("\t")
    ext1 = int(ll[1])-leftExt
    ext2 = int(ll[2])+rightExt
    if(ext1 >= ext2): continue
    bedTempFile.write("\t".join([ll[0],str(ext1),str(ext2)]+ll[3:])+"\n")
bedTempFile.close()
predFile.close()

# Empty prediction set
bedTempFile = open(outputFileName+"_temp.bed","r")
line = bedTempFile.readline()
bedTempFile.close()
if(line == ""):
    os.system("sort -k1,1 -k2,2n "+mpbsFileName+" | uniq > "+outputFileName)
    os.system("rm "+outputFileName+"_temp.bed")
    sys.exit(0)

# Getting number of columns in mpbs file
tempMpbsFile = open(mpbsFileName)
nColMpbs = len(tempMpbsFile.readline().strip().split("\t"))
tempMpbsFile.close()

# Sort prediction and mpbs bed files
os.system("sort -k1,1 -k2,2n "+mpbsFileName+" > "+outputFileName+"_mpbssort.bed")
os.system("sort -k1,1 -k2,2n "+outputFileName+"_temp.bed > "+outputFileName+"_sort.bed")

# Intersect with mpbs file
os.system("intersectBed -a "+outputFileName+"_mpbssort.bed -b "+outputFileName+"_sort.bed -wa -wb > "+outputFileName+"_i1.bed")
os.system("intersectBed -a "+outputFileName+"_mpbssort.bed -b "+outputFileName+"_sort.bed -wa -v > "+outputFileName+"_i2.bed")
os.system("rm "+outputFileName+"_temp.bed "+outputFileName+"_sort.bed "+outputFileName+"_mpbssort.bed")

# Merge bed files increasing the score of the MPBS with predictions
i1File = open(outputFileName+"_i1.bed","r")
i2File = open(outputFileName+"_i2.bed","r")
outputFile = open(outputFileName+"TEMP.bed","w")
for line in i1File:
    ll = line.strip().split("\t")
    ll[4] = str(int(float(ll[nColMpbs+4]))+1)
    outputFile.write("\t".join(ll[:nColMpbs])+"\n")
for line in i2File:
    ll = line.strip().split("\t")
    ll[4] = "0"
    outputFile.write("\t".join(ll)+"\n")

# Termination
i1File.close()
i2File.close()
outputFile.close()
os.system("sort -k1,1 -k2,2n "+outputFileName+"TEMP.bed | uniq > "+outputFileName)
os.system("rm "+outputFileName+"_i1.bed "+outputFileName+"_i2.bed "+outputFileName+"TEMP.bed")



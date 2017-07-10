#################################################################################################
# Apply PIQ [Gifford et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. pwnN = Number of the PWM to run.")
params.append("    2. factor = TF name.")
params.append("    3. pwmLoc = PATH to the motif matching generated by PIQ.")
params.append("    4. cutFileName = Rdata file with DNase cuts generated by PIQ.")
params.append("    5. outputLocation = Location of output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Bed file containing MPBSs with PIQ scores.")
params.append("###")
#################################################################################################

# Import
import os
import sys
from glob import glob
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
pwmN = sys.argv[1]
factor = sys.argv[2]
pwmLoc = sys.argv[3]
cutFileName = sys.argv[4]
outputLocation = sys.argv[5]

# Initialization
to_remove = []
outputFileName = outputLocation+"PIQ_"+factor+".bed"
tempLoc = outputLocation+"temp_"+factor+"/"
resLoc = outputLocation+"res_"+factor+"/"
to_remove.append(tempLoc)
to_remove.append(resLoc)
os.system("mkdir -p "+tempLoc)
os.system("mkdir -p "+resLoc)
l = "/home/eg474423/Installation/PIQ/"
cpVec = [l+e for e in ["bam2rdata.r","bed2pwm.r","bindcall.r","bindcall.standalone.r","cluster.r",
        "common.global.r","common.mm.r","common.r","covfun.r","loadbam.r","loadbam.sup.r","pairedbam2rdata.r",
        "pertf.bg.r","pertf.r","pertf.sup.r","pwmmatch.exact.r","pwmmatch.r"]]

os.system("cp "+" ".join(cpVec)+" "+tempLoc)

# Apply PIQ
os.system("cd "+tempLoc+";Rscript pertf.r common.r "+pwmLoc+" "+tempLoc+" "+resLoc+" "+cutFileName+" "+pwmN)

# Figuring motif length
motifLen = 1
bedFile = open(glob(resLoc+"*.bed")[0],"r")
bedFile.readline()
ll = bedFile.readline().strip().split("\t")
motifLen = int(ll[2])-int(ll[1])
bedFile.close()

# Fixing output
csvFileList = glob(resLoc+"*-calls.all.csv")
tempFileName = resLoc+"res.bed"
tempFile = open(tempFileName,"w")
for csvFileName in csvFileList:
  if(".RC-" in csvFileName): strand = "-"
  else: strand = "+"
  csvFile = open(csvFileName,"r")
  for line in csvFile:
    if("purity" in line): continue
    ll = line.strip().split(",")
    chrName = ll[1][1:-1]
    p1 = ll[2]
    p2 = str(int(ll[2])+motifLen)
    score = ll[5]
    tempFile.write("\t".join([chrName,p1,p2,factor,score,strand])+"\n")
tempFile.close()

# Sort
sortFileName = resLoc+"res_sort.bed"
os.system("sort -k1,1 -k2,2n "+tempFileName+" > "+sortFileName)

# Merge (keep max score)
mergeFileName = resLoc+"res_merge.bed"
os.system("/home/eg474423/Installation/bedtools2-master/bin/mergeBed -c 5 -o max -i "+sortFileName+" > "+mergeFileName)

# Creating final FP file
mergeFile = open(mergeFileName,"r")
outFile = open(outputFileName,"w")
counter = 1
for line in mergeFile:
  ll = line.strip().split("\t")
  outFile.write("\t".join(ll[:3]+["fp"+str(counter),ll[3],"."])+"\n")
  counter += 1
mergeFile.close()
outFile.close()

# Termination
os.system("rm -rf "+tempLoc+"*")
os.system("rm -rf "+resLoc+"*")
os.system("rm -rf "+tempLoc)
os.system("rm -rf "+resLoc)


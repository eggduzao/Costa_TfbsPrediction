#################################################################################################
# Apply Footprint Mixture [Ohler et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsFileName = BED with MPBSs.")
params.append("    2. peakFileName = BED with DNase peaks.")
params.append("    3. cutsFileName = TXT file with cuts.")
params.append("    4. seqFileName = FASTA with each MPBS's sequence.")
params.append("    5. outputFileName = Name of the output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputfileName> = Bed file containing MPBSs with Footprint Mixture scores.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsFileName = sys.argv[1]
peakFileName = sys.argv[2]
cutsFileName = sys.argv[3]
seqFileName = sys.argv[4]
outputFileName = sys.argv[5]

# Parameters
mvLoc = "/home/eg474423/app/FootprintMixture/"
mvFileList = [mvLoc+"MixtureModel.r", mvLoc+"RebuildSignal.pl", mvLoc+"SeqBias.txt"]

# Initialization
factor = mpbsFileName.split("/")[-1].split(".")[0]
outLoc = "/".join(outputFileName.split("/")[:-1])+"/"+factor+"/"
os.system("mkdir -p "+outLoc)
os.system("cp "+" ".join(mvFileList)+" "+outLoc)

# Creating R file to be executed
rFileName = outLoc+factor+".R"
tempFileName = outLoc+factor+".temp"
rFile = open(rFileName,"w")
rFile.write(".libPaths( c('/home/eg474423/R', .libPaths() ) )\n")
rFile.write("setwd('"+outLoc+"')\n")
rFile.write("library(GenomicRanges)\n")
rFile.write("library(gtools)\n")
rFile.write("library(mixtools)\n")
rFile.write("source('MixtureModel.r')\n")
rFile.write("c <- read.table('"+cutsFileName+"')\n")
rFile.write("b <- read.table('"+mpbsFileName+"')\n")
rFile.write("p <- read.table('"+peakFileName+"')\n")
rFile.write("m <- MultMMixture_Full(TF_Bed=b,Cuts=c,peakbed=p,Plot=F,PadLen=25,Collapse=T,k=2,ReturnPar=T,Fixed=T,Background='Seq',FastaName='"+seqFileName+"')\n")
rFile.write("write(m$llr, file = '"+tempFileName+"', ncolumns = 1, append = FALSE, sep = '')\n")
rFile.close()

# Running R script
flagRun = False
while(not flagRun):
  os.system("cd "+outLoc+";R CMD BATCH "+rFileName)
  try: 
    resFile = open(tempFileName,"r")
    resFile.close()
    flagRun = True
  except Exception: continue

# Fixing output
mpbsFile = open(mpbsFileName,"r")
resFile = open(tempFileName,"r")
outputFile = open(outputFileName,"w")
for line in mpbsFile:
  ll = line.strip().split("\t")
  v = resFile.readline().strip()
  outputFile.write("\t".join(ll[:4]+[v]+ll[5:])+"\n")
mpbsFile.close()
resFile.close()
outputFile.close()

# Termination
os.system("rm -rf "+outLoc)



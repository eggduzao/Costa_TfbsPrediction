#################################################################################################
# Apply dnase2tf [Hager et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. regionFileName = BED file with the regions to apply the algorithm.")
params.append("    2. dataFilePath = PATH with prefix to bed file cut count data.")
params.append("    3. mapFilePath = PATH to the mappability file.")
params.append("    4. dtfFileName = DTF file with dinucleotide frequencies.")
params.append("    5. genomePath = PATH to the genome split by chr.")
params.append("    6. outputFileName = Location + prefix of output file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputfileName> = Bed file containing MPBSs with dnase2tf scores.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
regionFileName = sys.argv[1]
dataFilePath = sys.argv[2]
mapFilePath = sys.argv[3]
dtfFileName = sys.argv[4]
genomePath = sys.argv[5]
outputFileName = sys.argv[6]

# Initialization
to_remove = []
outLoc = "/".join(outputFileName.split("/")[:-1])+"/"

# Creating new regionFile with only 3 columns
newRegionFileName = outLoc+"regions.bed"
os.system("cut -f 1,2,3 "+regionFileName+" > "+newRegionFileName)
to_remove = [newRegionFileName]

# Creating R file to be executed
rFileName = outputFileName+".R"
to_remove.append(rFileName)
rFile = open(rFileName,"w")
rFile.write(".libPaths( c('/home/eg474423/R', .libPaths() ) )\n")
rFile.write("library(dnase2tf)\n")
rFile.write("p1='"+newRegionFileName+"'\n")
rFile.write("p2='"+dataFilePath+"'\n")
rFile.write("p3='"+mapFilePath+"'\n")
rFile.write("p4='"+dtfFileName+"'\n")
rFile.write("p5='"+genomePath+"'\n")
rFile.write("p6='"+outputFileName+"'\n")
rFile.write("dnase2tf(p2, p1, p3, p6, assemseqdir=p5, dftfilename=p4, minw=6, maxw=30, z_threshold=-2, numworker=1, FDRs=0.001, biascorrection='tetramer')\n")
rFile.close()
bgrFileName = outputFileName+"_fdr0.001000.bgr"
outBedFileName = outputFileName+"_fdr0.001000.bed"
to_remove.append(bgrFileName)
to_remove.append(outBedFileName)

# Running R script
os.system("cd "+outLoc+";R CMD BATCH "+rFileName)
to_remove.append(rFileName+"out")
to_remove.append(outLoc+".RData")

# Creating footpyiny file
resFile = open(bgrFileName,"r")
resFile.readline()
outFile = open(outputFileName,"w")
counter = 1
for line in resFile:
  ll = line.strip().split("\t")
  outFile.write("\t".join(ll[:3]+["p"+str(counter),ll[3]])+"\n")
  counter += 1
resFile.close()
outFile.close()

# Termination
for e in to_remove: os.system("rm "+e)



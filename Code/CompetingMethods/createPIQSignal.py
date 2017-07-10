#################################################################################################
# Create DNase cut signal for PIQ [Gifford et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. bamFileName = BAM file with DNase-seq cuts.")
params.append("    2. outputFileName = Location of output files.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Rdata file with DNase signal.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
bamFileName = sys.argv[1]
outputFileName = sys.argv[2]

# Creating bam signal counts
os.system("cd /home/eg474423/Installation/PIQ/;Rscript bam2rdata.r common.r "+outputFileName+" "+bamFileName)



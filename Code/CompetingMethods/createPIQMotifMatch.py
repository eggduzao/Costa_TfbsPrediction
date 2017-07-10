#################################################################################################
# Create motif matching for PIQ [Gifford et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. pwnN = Number of the PWM to run.")
params.append("    2. pwmFileName = PATH with prefix to bed file cut count data.")
params.append("    3. outputLocation = Location of output files.")
params.append("###")
params.append("Output: ")
params.append("    1. <PWM files> = PIQ PWM files.")
params.append("###")
#################################################################################################

# Import
import os
import sys
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
pwnN = sys.argv[1]
pwmFileName = sys.argv[2]
outputLocation = sys.argv[3]

# Motif Matching
os.system("mkdir -p "+outputLocation)
os.system("cd /home/eg474423/Installation/PIQ/;Rscript pwmmatch.exact.r common.r "+pwmFileName+" "+pwnN+" "+outputLocation)



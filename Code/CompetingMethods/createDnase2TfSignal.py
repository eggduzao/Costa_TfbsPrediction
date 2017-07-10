#################################################################################################
# Create dnase2tf [Hager et al] signal
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. organism = 'hg19' or 'mm9'.")
params.append("    2. bamFileName = BAM file with the Dnase reads.")
params.append("    3. outputLocation = Location of output files.")
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
organism = sys.argv[1]
bamFileName = sys.argv[2]
outputLocation = sys.argv[3]

# Initialization
to_remove = []
if(organism == "hg19"): chrList = ["chr"+str(e) for e in range(1,23)+["X"]]
else: chrList = ["chr"+str(e) for e in range(1,20)+["X"]]

# Iterating on chromosomes
for chrName in chrList:

  # Split bam by chromosome
  bamChromFileName = outputLocation+chrName+".bam"
  os.system("samtools view -bh "+bamFileName+" "+chrName+" > "+bamChromFileName)

  # Convert to bed
  bedChromFileName = outputLocation+chrName+"_chrom.bed"
  os.system("bamToBed -i "+bamChromFileName+" > "+bedChromFileName)

  # Replace + by F and - by R
  bedCutFileName = outputLocation+"DNase_"+chrName+".txt"
  os.system("cut -f 1,2,3,6 "+bedChromFileName+" > "+bedCutFileName)
  os.system("sed -i 's/+/F/g;s/-/R/g' "+bedCutFileName)
  
  # Termination
  to_remove = [bamChromFileName,bedChromFileName]
  for e in to_remove: os.system("rm "+e)



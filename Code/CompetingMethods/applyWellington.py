#################################################################################################
# Apply Wellington [Ott et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. regionFileName = BED file with the regions to run the footprinting.")
params.append("    2. bamFileName = BAM file with DNase-seq reads.")
params.append("    3. outputFileName = Name of output BED file.")
params.append("###")
params.append("Output: ")
params.append("    1. <outputFileName> = Bed file containing Wellington footprints.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import pyDNase
import pyDNase.footprinting as fp
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
regionFileName = sys.argv[1]
bamFileName = sys.argv[2]
outputFileName = sys.argv[3]

# Parameters
cutoff = -30
footprintSizes = range(6,40,1)
to_remove = []

# Creating new region file name with the first three columns only
newRegionFileName = outputFileName+"regions.bed"
os.system("cut -f 1,2,3 "+regionFileName+"  > "+newRegionFileName)
to_remove.append(newRegionFileName)

# Execution
outputFile = open(outputFileName,"w")
regions = pyDNase.GenomicIntervalSet(newRegionFileName)
reads = pyDNase.BAMHandler(bamFileName)
for region in regions: 
  footprinter = fp.wellington(region,reads,shoulder_sizes=range(35,36), footprint_sizes = footprintSizes, FDR=0, bonferroni = 0)
  footprints = footprinter.footprints(withCutoff=cutoff)
  for e in footprints:
    outputFile.write("\t".join([str(k) for k in [e.chromosome, e.startbp, e.endbp, e.label, e.score , e.strand]])+"\n")
outputFile.close()

# Termination
for e in to_remove: os.system("rm "+e)



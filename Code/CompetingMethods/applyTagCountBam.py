
# Import
import os
import sys
from pysam import Samfile

# Reading input
totalWindow = int(sys.argv[1])
mpbsFileName = sys.argv[2]
bamFileName = sys.argv[3]
outputFileName = sys.argv[4]

# Creating bam file
bamFile = Samfile(bamFileName,"rb")

# Function to read pileup
def tag_count(chrName, start, end):
  total = 0
  for read in bamFile.fetch(reference=chrName, start=start, end=end): total += 1
  return total

# Iterating on the mpbsfile to update the score
halfWindow = totalWindow/2
mpbsFile = open(mpbsFileName,"r")
outputFile = open(outputFileName,"w")
for line in mpbsFile:
  ll = line.strip().split()
  mid = (int(ll[1])+int(ll[2]))/2
  p1 = max(mid - halfWindow,0)
  p2 = mid + halfWindow
  nCount = tag_count(ll[0], p1, p2)
  ll[4] = str(nCount) # ORIGINAL NATMET PAPER
  outputFile.write("\t".join(ll)+"\n") # ORIGINAL NATMET PAPER

# Termination
bamFile.close()
mpbsFile.close()
outputFile.close()



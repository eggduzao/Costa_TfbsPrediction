
# Import
import os
import sys
from pysam import Samfile

# Reading input
mpbsFileName = sys.argv[1]
bamFileName = sys.argv[2]
outputFileName = sys.argv[3]

# Creating bam file
bamFile = Samfile(bamFileName,"rb")

# Function to read pileup
def tag_count(chrName, start, end):
  total = 0
  for read in bamFile.fetch(reference=chrName, start=start, end=end): total += 1
  return total

# Iterating on the mpbsfile to update the score
mpbsFile = open(mpbsFileName,"r")
outputFile = open(outputFileName,"w")
for line in mpbsFile:
  ll = line.strip().split()
  chrName = ll[0]; p1 = int(ll[1]); p2 = int(ll[2])
  p1_ext = p1 - mLen; p2_ext = p2 + mLen
  if(p1_ext < 0): continue
  nl = tag_count(chrName, p1_ext, p1) + 1.0
  nc = tag_count(chrName, p1, p2) + 1.0
  nr = tag_count(chrName, p2, p2_ext) + 1.0
  try: ll[4] = str(round( -((nc/nr)+(nc/nl)) ,4))
  except Exception: ll[4] = "-999"
  outputFile.write("\t".join(ll)+"\n")

# Termination
bamFile.close()
mpbsFile.close()
outputFile.close()



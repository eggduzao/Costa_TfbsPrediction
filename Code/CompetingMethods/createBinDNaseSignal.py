#################################################################################################
# Create all files necessary for BinDNase software [Kahara et al]
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. mpbsFileName = BED file with the MPBSs.")
params.append("    2. bamFileName = BAM file with the reads.")
params.append("    3. outputLocation = Location of output file.")
params.append("###")
params.append("Output: ")
params.append("    1. [mpbs/pwm/dnase][train/test][pos/neg] = Output files for BinDNase.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import pysam as ps
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)

# Reading input
mpbsFileName = sys.argv[1]
bamFileName = sys.argv[2]
outputLocation = sys.argv[3]

# Parameters
ext = 220
toRemove = []
cell = bamFileName.split("/")[-2]
factor = mpbsFileName.split("/")[-1].split(".")[0]

# File Names
mpbsTrainPos = outputLocation+cell+"."+factor+".mpbs.train.positive"
mpbsTrainNeg = outputLocation+cell+"."+factor+".mpbs.train.negative"
mpbsTestPos = outputLocation+cell+"."+factor+".mpbs.test.positive"
mpbsTestNeg = outputLocation+cell+"."+factor+".mpbs.test.negative"
pwmTrainPos = outputLocation+cell+"."+factor+".pwm_scores.train.positive"
pwmTrainNeg = outputLocation+cell+"."+factor+".pwm_scores.train.negative"
pwmTestPos = outputLocation+cell+"."+factor+".pwm_scores.test.positive"
pwmTestNeg = outputLocation+cell+"."+factor+".pwm_scores.test.negative"
dnaseTrainPos = outputLocation+cell+"."+factor+".train.positive"
dnaseTrainNeg = outputLocation+cell+"."+factor+".train.negative"
dnaseTestPos = outputLocation+cell+"."+factor+".test.positive"
dnaseTestNeg = outputLocation+cell+"."+factor+".test.negative"

# Pileup class
class PileupRegion:
  def __init__(self,start,end,ext):
    self.start = start
    self.end = end
    self.length = end-start
    self.ext = ext
    self.vector = [0.0] * self.length
  def __call__(self, alignment):
    if(not alignment.is_reverse):
      for i in range(max(alignment.pos,self.start),min(alignment.pos+self.ext,self.end-1)):
        self.vector[i-self.start] += 1.0
    else:
      if(alignment.aend):
        for i in range(max(alignment.aend-self.ext,self.start),min(alignment.aend,self.end-1)):
          self.vector[i-self.start] += 1.0

# Function to create cuts file
def createCuts(ext, mpbsFileName, bamFileName, outputFileName):
  mpbsFile = open(mpbsFileName,"r")
  bamFile = ps.AlignmentFile(bamFileName, "rb")
  cutFile = open(outputFileName,"w")
  for line in mpbsFile:
    ll = line.strip().split("\t")
    if(int(ll[1])-ext <= 0): continue
    pileup_region = PileupRegion(int(ll[1])-ext,int(ll[2])+ext,1)
    iter = bamFile.fetch(reference=ll[0], start=int(ll[1])-ext, end=int(ll[2])+ext) 
    for alignment in iter: pileup_region.__call__(alignment) 
    cutFile.write("\t".join([str(int(e)) for e in pileup_region.vector])+"\n")
  bamFile.close()
  mpbsFile.close()
  cutFile.close()


# Separate train and test / neg and pos
trainTemp = outputLocation+cell+"."+factor+".train"; toRemove.append(trainTemp)
testTemp = outputLocation+cell+"."+factor+".test"; toRemove.append(testTemp)
os.system("grep chr1\"\t\" "+mpbsFileName+" > "+trainTemp)
os.system("grep -v chr1\"\t\" "+mpbsFileName+" > "+testTemp)
os.system("grep :Y "+trainTemp+" > "+mpbsTrainPos)
os.system("grep :N "+trainTemp+" > "+mpbsTrainNeg)
os.system("grep :Y "+testTemp+" > "+mpbsTestPos)
os.system("grep :N "+testTemp+" > "+mpbsTestNeg)

# Fetching PWM scores
os.system("cut -f 5 "+mpbsTrainPos+" > "+pwmTrainPos)
os.system("cut -f 5 "+mpbsTrainNeg+" > "+pwmTrainNeg)
os.system("cut -f 5 "+mpbsTestPos+" > "+pwmTestPos)
os.system("cut -f 5 "+mpbsTestNeg+" > "+pwmTestNeg)

# Creating cuts files
createCuts(ext, mpbsTrainPos, bamFileName, dnaseTrainPos)
createCuts(ext, mpbsTrainNeg, bamFileName, dnaseTrainNeg)
createCuts(ext, mpbsTestPos, bamFileName, dnaseTestPos)
createCuts(ext, mpbsTestNeg, bamFileName, dnaseTestNeg)

# Termination
for e in toRemove: os.system("rm "+e)



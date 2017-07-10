#################################################################################################
# Runs the pipeline for the EXTREME algorithm.
#
# The following are arguments for GappedKmerSearch.py, the word searching algorithm for the seeding:
#   -l HALFLENGTH. The number of exact letters on each side of the word (default 4).
#   -ming MINGAP. The minimum number of universal wildcard letters in the middle (default 0).
#   -maxg MAXGAP. The maximum number of universal wildcard letters in the middle (default 10).
#   -minsites MINSITES. Minimum number of sites a word should have to be included (default 10).
#   -zthresh ZTHRESHOLD. Minimum normalized z-score for a word to be saved. A lower threshold increases
#                        the number of words saved (default 5).
#
# The following are arguments for run_consensus_clusering_using_wm.pl, the hierarchical clustering algorithm for the seeding:
#   THRESHOLD. The threshold for the clustering. Has values between 0 and 1. A value closer to 1
#              decreases the number of clusters, while a value closer to 0 increases the number
#              of clusters. Recommended value is 0.3.
#
# The following are arguments for EXTREME.py, the EXTREME algorithm:
#   -t TRIES. The number of different bias factors to try before giving up on the current seed.
#   -s SEED. Random seed for shuffling sequences and dataset positions.
#   -p PSEUDOCOUNTS. Uniform pseudo counts to add to initial PFM guess (default 0.0).
#   -q INITALSTEP. The initial step size for the online EM algorithm. A VERY sensitive parameter.
#                  I get best success for ChIP size data (about 100,000 to 1,000,000 bps) with a step
#                  size of 0.05. For DNase footprinting, which usually has >5,000,000 bps, I find 0.02
#                  works best (default 0.05).
#   -minsites MINSITES. Minimum number of sites the motif should have (default 10).
#   -maxsites MAXSITES. Minimum number of sites the motif should have. If not specified, it is set to
#                       five times the number of predicted motif sites based on the initial PFM guess
#   -saveseqs SAVESEQS. A switch. If used, the positive and negative sequence set will be saved to
#                       Positive_seq.fa and Negative_seq.fa, respectively, with instances of the
#                       discovered motif replaced with capital Ns.
#################################################################################################
params = []
params.append("###")
params.append("Input: ")
params.append("    1. bedFileName = Name of the input bed file.")
params.append("    2. genomeFileName = Genome to extract the sequences.")
params.append("    3. outputLocation = Location of the output and temporary files.")
params.append("###")
params.append("Output: ")
params.append("    1. EXTREME output.")
params.append("###")
#################################################################################################

# Import
import os
import sys
import glob
import re
lib_path = os.path.abspath("/".join(os.path.realpath(__file__).split("/")[:-2]))
sys.path.append(lib_path)
from util import *
if(len(sys.argv) <= 1): 
    for e in params: print e
    sys.exit(0)
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

###################################################################################################
# INPUT
###################################################################################################

# Reading input
bedFileName = sys.argv[1]
genomeFileName = sys.argv[2]
outputLocation = sys.argv[3]
if(outputLocation[-1] != "/"): outputLocation+="/"

# Parameters for GappedKmerSearch
halfSiteLength = "8"
ming = "0"
maxg = "10"
minsites = "5"

# Parameters for run_consensus_clusering_using_wm
clusterThresh = "0.3"

# Creating fasta file
inName = ".".join(bedFileName.split("/")[-1].split(".")[:-1])
fastaFileName = outputLocation+inName+".fasta"
toRemove = [fastaFileName]
os.system("fastaFromBed -fi "+genomeFileName+" -bed "+bedFileName+" -fo "+fastaFileName)


###################################################################################################
# EXTREME
###################################################################################################

loc = "/home/egg/Desktop/footprint_motifmatch/Extreme/EXTREME-2.0.0/src/"

# 1. Generates a dinucleotide shuffled version of the positive sequence set to serve as a negative sequence set

shuffledFileName = outputLocation+inName+"_shuffled.fasta"
toRemove.append(shuffledFileName)
os.system("python "+loc+"fasta-dinucleotide-shuffle.py -f "+fastaFileName+" > "+shuffledFileName)

# 2. Finds gapped words with two half-sites of length "-l", between "-ming" and "-maxg" universal wildcard gap letters, and at least "-minsites" occurrences in the positive sequence set

wordsFileName = outputLocation+inName+".words"
toRemove.append(wordsFileName)
os.system("python "+loc+"GappedKmerSearch.py -l "+halfSiteLength+" -ming "+ming+" -maxg "+maxg+" -minsites "+minsites+" "+fastaFileName+" "+shuffledFileName+" "+wordsFileName)

# 3. Clusters the words and outputs the results

clusterFileName = wordsFileName+".cluster.aln"
toRemove.append(clusterFileName)
os.system("perl "+loc+"run_consensus_clusering_using_wm.pl "+wordsFileName+" "+clusterThresh)

# 4. Converts the clusters into PFMs which can be used as seeds for the online EM algorithm

wmFileName = outputLocation+inName+".wm"
toRemove.append(wmFileName)
os.system("python "+loc+"Consensus2PWM.py "+clusterFileName+" "+wmFileName)
nbClusters = 0
wmFile = open(wmFileName,"r")
for line in wmFile:
  if(line[0] == ">"): nbClusters += 1
wmFile.close()

# 5. EM algorithm

memePfmFileName = outputLocation+"all_pwms.meme"
memePfmFile = open(memePfmFileName,"w")
memePfmFile.write("MEME version 4.9.0\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies (from uniform background):\nA 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n")

htmlFileName = outputLocation+"all_motifs.html"
htmlFile = open(htmlFileName,"w")
htmlFile.write("<html>\n<head></head>\n<body>\n")

for i in range(1,nbClusters+1):
  os.system("python "+loc+"EXTREME.py "+fastaFileName+" "+shuffledFileName+" "+wmFileName+" "+str(i))
  memeFile = open("./cluster"+str(i)+"/MEMEoutput.meme","r")
  flagStart = False
  for line in memeFile:
    if(len(line) > 5 and line[:5] == "MOTIF"): flagStart = True
    if(flagStart): memePfmFile.write(line)
  htmlFile.write("<h1>cluster"+str(i)+"</h1>\n")
  htmlFile.write("<table cellpadding=\"4\" style=\"border: 1px solid #000000; border-collapse: collapse;\" border=\"1\">\n")
  pngList = natural_sort(glob.glob(outputLocation+"cluster"+str(i)+"/Motif*.png"))
  for pngFileName in pngList:
      mname = ".".join(pngFileName.split("/")[-1].split(".")[:-1])
      htmlFile.write("<tr><td>"+mname+"</td><td><img src=\""+pngFileName+"\"></td></tr>\n")
  htmlFile.write("</table>\n")
  memeFile.close()

memePfmFile.close()
htmlFile.close()

###################################################################################################
# OUTPUT
###################################################################################################

# Remove temporary files
for e in toRemove: os.system("rm "+e)



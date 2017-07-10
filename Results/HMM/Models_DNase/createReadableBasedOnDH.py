
# Import
import os
import sys

# Iterating on cell types
cellList = ["H1hesc","HeLaS3","HepG2","Huvec","K562"]
for cell in cellList:

  # Initialization
  inFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNaseHistone/"+cell+"/Annotation/readable/H3K4me3_proximal.txt"
  outFileName = "/home/egg/Projects/TfbsPrediction/Results/HMM/Models_DNase/"+cell+"/Annotation/dnase2.txt"
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")

  # Writing header
  outFile.write("\n# Coordinates are 1-relative (Genome browser and wig files)\n")
  outFile.write("0 = BACK\n")
  outFile.write("1 = UPD\n")
  outFile.write("2 = TOPD\n")
  outFile.write("3 = DOWND\n")
  outFile.write("4 = FP\n\n")

  # Writing states
  counter = 0
  for line in inFile:
    counter += 1
    if("#" in line or "=" in line or len(line) < 2):
      if(counter > 11): outFile.write("\n")
      continue
    ll = line.strip().split(" ")
    newseqs = []
    for seq in ll[1:-1]:
      newseq = []
      for e in seq:
        if(e in ['0','1','2','3']): newseq.append("0")
        elif(e == '4'): newseq.append("1")
        elif(e == '5'): newseq.append("2")
        elif(e == '6'): newseq.append("3")
        elif(e == '7'): newseq.append("4")
      newseqs.append("".join(newseq))
    outFile.write(" ".join([ll[0]]+newseqs+[ll[-1]])+"\n")
  inFile.close()
  outFile.close()



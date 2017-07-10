
# Import
import os
import sys
from glob import glob
import sys

# Parameters
tVec = ["80", "85", "90", "95", "99"]
centipedeThresholds = ["0.80", "0.85", "0.90", "0.95", "0.99"]
cuellarThresholds = ["1.93878", "2.26531", "2.59184", "3.26531", "4.79592"]
flrThresholds = ["3.3765732", "4.77054665", "7.1567691", "11.826144", "30.6471376"]
piqThresholds = ["5.693901445", "6.2552475406", "6.9880273606", "8.0436170598", "15.8327019228"]
flagCentipede = True
flagCuellar = True
flagFLR = True
flagPIQ = True
hsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/All/mergedHS.bed"

def file_len(fname):
  i = 0
  with open(fname) as f:
    for i, l in enumerate(f): pass
  return i + 1

# Cell Loop
cellList = ["GM12878", "H1hesc", "K562"]
for cell in cellList:

  # Threshold
  for t in range(0,len(tVec)):

    tLabel = tVec[t]

    #####################
    # Centipede
    #####################

    if(flagCentipede):

      thresh = centipedeThresholds[t]
      toRemove = []
      inLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Centipede_batch/"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"
      inList = glob(inLoc+"*.bed")
      for fn in inList:
        print "\t".join([cell,tLabel,"Centipede",fn.split("/")[-1].split(".")[0]])
        sys.stdout.flush()
        tempFn = fn+"T"; toRemove.append(tempFn)
        os.system("awk '$5>="+thresh+"' "+fn+" | cut -f 1,2,3 > "+tempFn)
        if(file_len(tempFn) <= 1):
          lenB = file_len(tempFn)
          os.system("cut -f 1,2,3 "+fn+" > "+tempFn)
          print "\t".join(["Empty file problem: ",str(lenB),str(file_len(tempFn))])
      outFs = outLoc+"centipedesort.bed"
      outFn = outLoc+"Centipede_"+tLabel+"_batch.bed"
      os.system("cat "+" ".join(toRemove)+" | sort -k1,1 -k2,2n > "+outFs)
      os.system("mergeBed -i "+outFs+" > "+outFn)
      toRemove.append(outFs)
      for e in toRemove: os.system("rm "+e)

    #####################
    # Cuellar
    #####################

    if(flagCuellar):

      thresh = cuellarThresholds[t]
      toRemove = []
      hsList = []
      inLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/Cuellar_batch/"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"
      inList = glob(inLoc+"*.bed")
      for fn in inList:
        print "\t".join([cell,tLabel,"Cuellar",fn.split("/")[-1].split(".")[0]])
        sys.stdout.flush()
        tempFn = fn+"T"; toRemove.append(tempFn)
        tempHs = fn+"HS.bed"; toRemove.append(tempHs); hsList.append(tempHs)
        os.system("awk '$5>="+thresh+"' "+fn+" | awk -v OFS='\\t' '{split($1,a,\":\"); print a[1],$2,$3}' > "+tempFn)
        if(file_len(tempFn) <= 1):
          lenB = file_len(tempFn)
          os.system("awk -v OFS='\\t' '{split($1,a,\":\"); print a[1],$2,$3}' "+fn+" > "+tempFn)
          print "\t".join(["Empty file problem: ",str(lenB),str(file_len(tempFn))])
        os.system("intersectBed -wa -u -a "+tempFn+" -b "+hsFileName+" > "+tempHs)
      outFs = outLoc+"cuellarsort.bed"
      outFn = outLoc+"Cuellar_"+tLabel+"_batch.bed"
      os.system("cat "+" ".join(hsList)+" | sort -k1,1 -k2,2n > "+outFs)
      os.system("mergeBed -i "+outFs+" > "+outFn)
      toRemove.append(outFs)
      for e in toRemove: os.system("rm "+e)

    #####################
    # FLR
    #####################

    if(flagFLR):

      thresh = flrThresholds[t]
      toRemove = []
      inLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/FLR_batch/"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"
      inList = glob(inLoc+"*.bed")
      for fn in inList:
        print "\t".join([cell,tLabel,"FLR",fn.split("/")[-1].split(".")[0]])
        sys.stdout.flush()
        tempFn = fn+"T"; toRemove.append(tempFn)
        os.system("awk '$5>="+thresh+"' "+fn+" | cut -f 1,2,3 > "+tempFn)
        if(file_len(tempFn) <= 1):
          lenB = file_len(tempFn)
          os.system("cut -f 1,2,3 "+fn+" > "+tempFn)
          print "\t".join(["Empty file problem: ",str(lenB),str(file_len(tempFn))])
      outFs = outLoc+"flrsort.bed"
      outFn = outLoc+"FLR_"+tLabel+"_batch.bed"
      os.system("cat "+" ".join(toRemove)+" | sort -k1,1 -k2,2n > "+outFs)
      os.system("mergeBed -i "+outFs+" > "+outFn)
      toRemove.append(outFs)
      for e in toRemove: os.system("rm "+e)

    #####################
    # PIQ
    #####################

    if(flagPIQ):

      thresh = piqThresholds[t]
      toRemove = []
      hsList = []
      inLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/PIQ_batch/"
      outLoc = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions/"+cell+"/"
      inList = glob(inLoc+"*.bed")
      for fn in inList:
        print "\t".join([cell,tLabel,"PIQ",fn.split("/")[-1].split(".")[0]])
        sys.stdout.flush()
        tempFn = fn+"T"; toRemove.append(tempFn)
        tempHs = fn+"HS.bed"; toRemove.append(tempHs); hsList.append(tempHs)
        os.system("awk '$5>="+thresh+"' "+fn+" | cut -f 1,2,3 > "+tempFn)
        if(file_len(tempFn) <= 1):
          lenB = file_len(tempFn)
          os.system("cut -f 1,2,3 "+fn+" > "+tempFn)
          print "\t".join(["Empty file problem: ",str(lenB),str(file_len(tempFn))])
        os.system("intersectBed -wa -u -a "+tempFn+" -b "+hsFileName+" > "+tempHs)
      outFs = outLoc+"piqsort.bed"
      outFn = outLoc+"PIQ_"+tLabel+"_batch.bed"
      os.system("cat "+" ".join(hsList)+" | sort -k1,1 -k2,2n > "+outFs)
      os.system("mergeBed -i "+outFs+" > "+outFn)
      toRemove.append(outFs)
      for e in toRemove: os.system("rm "+e)

print "DONE!"
sys.stdout.flush()



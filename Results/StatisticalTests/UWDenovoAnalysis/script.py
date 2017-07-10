# Import
import os
import sys

# Input
hintFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/H7hesc/HMM_DNase_UW_BIAS_ModelH7hesc.bed"
nephFileName = "/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Predictions_NM/H7hesc/Neph_UW.bed"
dhsFileName = "/hpcwork/izkf/projects/TfbsPrediction/Data/DNaseUW/H7hesc/DNasePeaks.bed"
ml = "/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/H7hesc_Evidence/fdr_4/"
mpbsFileList = [ml+"UW_Motif_0458.bed", ml+"UW_Motif_0500.bed"]
tempLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/UWDenovoAnalysis/"

# Execution
for mpbsFileName in mpbsFileList:
  mpbsName = mpbsFileName.split("/")[-1]
  print mpbsName+"\n"
  sys.stdout.flush()
  newMpbsFileName = tempLoc+mpbsName
  os.system("intersectBed -wa -a "+mpbsFileName+" -b "+dhsFileName+" > "+newMpbsFileName)
  print "Total MPBS:"
  sys.stdout.flush()
  os.system("wc -l "+newMpbsFileName)
  print "Neph Intersection:"
  sys.stdout.flush()
  os.system("intersectBed -wa -a "+newMpbsFileName+" -b "+nephFileName+" | wc -l ")
  sys.stdout.flush()
  print "Total MPBS:"
  sys.stdout.flush()
  os.system("wc -l "+mpbsFileName)
  newHintFileName = tempLoc+mpbsName+"_hint.bed"
  os.system("intersectBed -u -wa -a "+hintFileName+" -b "+dhsFileName+" > "+newHintFileName)
  print "HINT Intersection:"
  sys.stdout.flush()
  os.system("intersectBed -u -wa -a "+newMpbsFileName+" -b "+newHintFileName+" | wc -l ")
  sys.stdout.flush()
  print "---------\n"
  sys.stdout.flush()
  os.system("rm "+newMpbsFileName+" "+newHintFileName)



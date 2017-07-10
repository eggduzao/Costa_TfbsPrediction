
# Import
import os
import sys
from Bio import motifs
from glob import glob

# Reading input
inputLoc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
outLoc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/TFBiasGraphs/Methods_Counts/Figure/"
inList = [ "DU_H1hesc" ]
factorList = ["EGR1","NRF1"]
intDict = dict([("EGR1",(22,31)),("NRF1",(21,31))])
showX = False
showY = False

# Function to cut pwm to make logo
# p1 and p2 are 1-based and both are included in the interval
def cutPwm(p1,p2,inFileName,outFileName):
  inFile = open(inFileName,"r")
  outFile = open(outFileName,"w")
  for line in inFile:
    ll = line.strip().split(" ")
    outFile.write(" ".join(ll[p1-1:p2])+"\n")
  inFile.close()
  outFile.close()

# Function to create logo based on pwm file
def createLogo(inFileName,outFileName,showX,showY):
  inFile = open(inFileName,"r")
  pwm = motifs.read(inFile, "pfm")
  pwm.weblogo(outFileName, format="eps", stack_width = "medium", stacks_per_line = "50", color_scheme = "color_classic", show_xaxis = showX, show_yaxis = showY, show_fineprint = False)
  inFile.close()
  os.system("epstopdf "+outFileName)

# Execution
for foldName in inList:

  for inFileName in glob(inputLoc+foldName+"/PWM/*_align.pwm"):

    # Initialization
    to_remove = []
    inName = inFileName.split("/")[-1].split("_")[0]
    if(inName not in factorList): continue

    # Creating [-15,15] logo file
    p1 = 11; p2 = 41
    tempInputFileName1 = outLoc+"temp1.pwm"
    to_remove.append(tempInputFileName1)
    cutPwm(p1,p2,inFileName,tempInputFileName1)
    if(showX): outFileName = outLoc+inName+"_PWM_withLabel.eps"
    else: outFileName = outLoc+inName+"_PWM_woLabel.eps"
    createLogo(tempInputFileName1,outFileName,showX,showY)

    # Creating zoom logo file
    p1 = intDict[inName][0]; p2 = intDict[inName][1]
    tempInputFileName2 = outLoc+"temp2.pwm"
    to_remove.append(tempInputFileName2)
    cutPwm(p1,p2,inFileName,tempInputFileName2)
    if(showX): outFileName = outLoc+inName+"_PWM_zoom_withLabel.eps"
    else: outFileName = outLoc+inName+"_PWM_zoom_woLabel.eps"
    createLogo(tempInputFileName2,outFileName,showX,showY)

    os.system("rm "+" ".join(to_remove))



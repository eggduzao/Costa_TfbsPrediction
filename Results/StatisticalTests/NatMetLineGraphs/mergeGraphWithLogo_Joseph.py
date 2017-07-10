
# Import
import os
import sys
from glob import glob
from pyx import *
from pyx.bbox import bbox

# Input
lineLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NatMetLineGraphs/Graphs/"
pwmLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"
inList = ["DU_H1hesc"]

# Iterating on each folder
for inFolder in inList:

  # Removing previous results because of the glob
  os.system("rm "+lineLoc+inFolder+"/*_pwm.eps")

  # Iterating on all TFs
  for lineFileName in glob(lineLoc+inFolder+"/*.eps"):

    # File names
    factorName = lineFileName.split("/")[-1].split(".")[0]
    pwmFileName = pwmLoc+inFolder+"/PWM/"+factorName+"_align.eps"
    #pwmFileName = pwmLoc+inFolder+"/PWM/"+factorName+".eps"
    outputFileName = lineFileName[:-4]+"_joseph.eps"

    b = bbox(1.3,0.725,21,3)

    # Creating canvas and printing eps / pdf with merged results
    c = canvas.canvas()
    #c.insert(epsfile.epsfile(2.12, 3, pwmFileName, scale=0.707))
    c.insert(epsfile.epsfile(1.91, 1.36, pwmFileName, width=15.4,height=1.68, bbox=b))
    #c.insert(epsfile.epsfile(0, 0, lineFileName, scale=1.0))
    c.writeEPSfile(outputFileName)
    os.system("epstopdf "+outputFileName)



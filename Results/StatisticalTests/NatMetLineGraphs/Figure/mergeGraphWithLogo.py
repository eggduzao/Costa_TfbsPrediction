
# Import
import os
import sys
from glob import glob
from pyx import *
from pyx.bbox import bbox

# Input
lineLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/NatMetLineGraphs/Figure/"
pwmLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/Pearson/"

# Removing previous results because of the glob
os.system("rm "+lineLoc+"*_pwm.eps")

# Iterating on all TFs
for lineFileName in glob(lineLoc+"*.eps"):

    # File names
    factorName = lineFileName.split("/")[-1].split(".")[0]
    pwmFileName = pwmLoc+"DU_H1hesc/PWM/"+factorName+"_align.eps"
    if("40min" in lineFileName): pwmFileName = pwmLoc+"DU_Mcf7/PWM/"+factorName+"_align.eps"
    outputFileName = lineFileName[:-4]+"_pwm.eps"

    b = bbox(1.3,0.725,21,3)

    # Creating canvas and printing eps / pdf with merged results
    c = canvas.canvas()
    c.insert(epsfile.epsfile(1.02, 0.5, pwmFileName, width=13.9,height=2.4, bbox=b))
    c.insert(epsfile.epsfile(0, 0, lineFileName, scale=1.0))
    c.writeEPSfile(outputFileName)
    os.system("epstopdf "+outputFileName)




# Import
import os
import sys
from glob import glob
from pyx import *
from pyx.bbox import bbox

# Input
lineLoc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/TFBiasGraphs/Methods_Counts/Figure2/"
pwmLoc = "/home/egg/Projects/TfbsPrediction/Results/StatisticalTests/TFSpecificBias/TFBiasGraphs/Methods_Counts/Figure2/"
inList = ["DU_H1hesc"]

# Iterating on each folder
for inFolder in inList:

  # Removing previous results because of the glob
  os.system("rm "+lineLoc+"*_pwm.eps")

  # Iterating on all TFs
  for lineFileName in [lineLoc+"EGR1_woLabel.eps", lineLoc+"NRF1_woLabel.eps"]:

    # File names
    factorName = lineFileName.split("/")[-1].split("_")[0]
    if(factorName == "EGR1"):
      myHeight = 3.4
      myY = 2.23
    else:
      myHeight = 3.4
      myY = 2.23
    pwmFileName = pwmLoc+factorName+"_PWM_woLabel.eps"
    outputFileName = lineFileName[:-4]+"_pwm.eps"

    # Creating canvas and printing eps / pdf with merged results
    c = canvas.canvas()
    c.insert(epsfile.epsfile(2.788, myY, pwmFileName, width=11.55,height=myHeight))
    c.insert(epsfile.epsfile(0, 0, lineFileName, scale=1.0))
    c.writeEPSfile(outputFileName)
    os.system("epstopdf "+outputFileName)



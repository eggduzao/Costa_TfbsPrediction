
# Import
import os
import sys
from glob import glob
from pyx import *
from pyx.bbox import bbox

# Input
lineLoc = "/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/"
pwmLoc = "/home/egg/Projects/TfbsPrediction/Report/regGenSIG2015/Figure/"
inList = ["DU_K562"]

# Iterating on each folder
for inFolder in inList:

  # Removing previous results because of the glob
  os.system("rm "+lineLoc+"*_pwm.eps")

  # Iterating on all TFs
  for lineFileName in [lineLoc+"CTCFL_woLabel.eps", lineLoc+"ZBTB7A_woLabel.eps"]:

    # File names
    factorName = lineFileName.split("/")[-1].split("_")[0]
    if(factorName == "CTCFL"):
      myHeight = 3.4
      myY = 2.23
    else:
      myHeight = 3.4
      myY = 2.23
    pwmFileName = pwmLoc+factorName+"_PWM_woLabel.eps"
    outputFileName = lineFileName[:-4]+"_pwm.eps"

    # Creating canvas and printing eps / pdf with merged results
    c = canvas.canvas()
    c.insert(epsfile.epsfile(2.795, myY, pwmFileName, width=12.4,height=myHeight))
    c.insert(epsfile.epsfile(0, 0, lineFileName, scale=1.0))
    c.writeEPSfile(outputFileName)
    os.system("epstopdf "+outputFileName)



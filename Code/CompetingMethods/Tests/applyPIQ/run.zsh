#!/bin/zsh

pwnN="139"
factor="CTCF"
pwmLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyPIQ/pwms/"
cutFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyPIQ/k562.RData"
outputLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyPIQ/"

applyPIQ $pwnN $factor $pwmLoc $cutFileName $outputLocation



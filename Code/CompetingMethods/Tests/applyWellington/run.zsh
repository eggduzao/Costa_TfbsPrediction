#!/bin/zsh
regionFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyWellington/DNasePeaks.bed"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/K562/DNase.bam"
outputFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/CompetingMethods/Tests/applyWellington/fp.bed"
applyWellington $regionFileName $bamFileName $outputFileName

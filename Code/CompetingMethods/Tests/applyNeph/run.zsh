#!/bin/zsh

fosThresh="0.95"
coordFileName="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/4.Footprint/Tests/applyNeph/region.bed"
dnaseFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/H1hesc/DNase.bw"
outputFileName="./footprints.bed"

applyNeph $fosThresh $coordFileName $dnaseFileName $outputFileName



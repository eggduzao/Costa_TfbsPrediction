#!/bin/zsh

totalWindow="200"
bedFileNameList="./CTCF_false.bed,./CTCF_true.bed"
bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/ATAC/K562/ATAC.bam"
outFileName="./profile.txt"
signalExt="2"
signalExtBoth="Y"

profileGraph $totalWindow $bedFileNameList $bamFileName $outFileName $signalExt $signalExtBoth



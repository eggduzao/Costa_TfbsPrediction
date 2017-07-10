#!/bin/zsh

# Parameters
genomeLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/"
genomeName="hg19.fa"

# Execution
bsub -J "BWAI" -o "BWAI_out.txt" -e "BWAI_err.txt" -W 48:00 -M 24000 -S 100 -R "select[hpcwork]" ./bwaIndex_pipeline.zsh $genomeLoc $genomeName
# -P izkf


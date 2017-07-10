#!/bin/zsh

inputList=( 
"/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"*"_Evidence/fdr_4/"*".bed"
)

# Execution
for inputFile in $inputList
do
  echo $inputFile","
  grep ":Y" $inputFile | wc -l
done


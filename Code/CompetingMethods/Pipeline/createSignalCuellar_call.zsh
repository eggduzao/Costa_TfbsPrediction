#!/bin/zsh

# Global parameters
chromSizesFile="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"

# Cell Line Parameters
#cellList=( "H1hesc" "K562" "GM12878" )
cellList=( "GM12878" )

# Cell Line Loop
for cell in $cellList
do

  # Signal Parameters
  bamFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_DU/"$cell"/DNase.bam"
  outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/DU_"$cell"/Cuellar/"
  #folderList=( "DNase" "Histone" "Histone" "Histone" "Histone" "Histone" )
  #signalList=( "DNase" "H3K4me1" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
  #minList=( "20" "25" "25" "25" "25" "25" )
  #maxList=( "65" "275" "275" "275" "275" "275" )
  #quartList=( "0" "100" "100" "100" "100" "100" )
  mkdir -p $outputLocation

  folderList=( "DNase" )
  signalList=( "DNase" )
  minList=( "20" )
  maxList=( "65" )
  quartList=( "0" )

  # Signal Loop
  for i in {1..$#folderList}
  do

    # Execution
    bsub -J $cell"_"$signalList[$i]"_SGC" -o $cell"_"$signalList[$i]"_SGC_out.txt" -e $cell"_"$signalList[$i]"_SGC_err.txt" -W 100:00 -M 80000 -S 100 -P izkf -R "select[hpcwork]" ./createSignalCuellar_pipeline.zsh $minList[$i] $maxList[$i] $quartList[$i] $chromSizesFile $bamFileName $outputLocation

  done
done



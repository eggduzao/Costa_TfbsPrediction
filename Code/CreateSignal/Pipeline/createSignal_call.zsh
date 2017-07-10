#!/bin/zsh

# Global Parameters
signalLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/Counts/"
coordLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/HMM/InputRegions/"
outLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/"

# Cell Line Parameters
#cellList=( "DU_H1hesc" "DU_H7hesc" "DU_HeLaS3" "DU_HepG2" "DU_Huvec" "DU_K562" "UW_HepG2" "UW_Huvec" "UW_K562" "DU_Mcf7" "UW_LnCaP" "UW_m3134_with_DEX" "UW_m3134_wo_DEX" "UW_H7hesc" )
cellList=( "DU_GM12878" )

# Cell Line Loop
for cell in $cellList
do

  # Coordinate Parameters
  coordList=( "dnase_HS_regions" )

  # Coordinate Loop
  for coord in $coordList
  do
  
    # Signal Parameters
    #signalList=( "DNase" "H3K4me1" "H3K4me3" "H2AZ" "H3K9ac" "H3K27ac" )
    #slopeWindowSizeList=( "9" "201" "201" "201" "201" "201" )
    #perNormList=( "98" "98" "98" "98" "98" "98" )
    #perSlopeList=( "98" "98" "98" "98" "98" "98" )
    signalList=( "DNase" )
    slopeWindowSizeList=( "9" )
    perNormList=( "98" )
    perSlopeList=( "98" )

    # Signal Loop
    for i in {1..$#signalList}
    do

      # Parameters
      slopeWindowSize=$slopeWindowSizeList[$i]
      perNorm=$perNormList[$i]
      perSlope=$perSlopeList[$i]
      coordFileName=$coordLocation$cell"/"$coord".bed"
      signalFileName=$signalLocation$cell"/"$signalList[$i]".bw"
      outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Signals/NormSlope/"$cell"/"
      mkdir -p $outputLocation

      # Execution
      bsub -J "NS" -o "NS_out.txt" -e "NS_err.txt" -W 24:00 -M 30000 -S 100 -P izkf -R "select[hpcwork]" ./createSignal_pipeline.zsh $slopeWindowSize $perNorm $perSlope $coordFileName $signalFileName $outputLocation
      # 

    done
  done
done



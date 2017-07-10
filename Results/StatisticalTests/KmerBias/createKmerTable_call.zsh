#!/bin/zsh

# Output Type Parameters
#typeList=( "HS" "All" )
typeList=( "HS" "All" )

# Output Type Loop
for type in $typeList
do

  # Input Parameters
  inputLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerDict/"$type"/"
  normLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/NormFactor/"
  outputLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerTable/"$type"/"
  #inList=( "DNase_H1hesc" "DNaseUW_H7hesc" "DNase_HeLaS3" "DNase_HepG2" "DNase_Huvec" "DNase_K562" "DNaseUW_HepG2" "DNaseUW_Huvec" "DNaseUW_K562" "DNase_Mcf7" "DNaseUW_IMR90" "DNaseUW_LnCaP" "DNaseUW_m3134_with_DEX" "DNaseUW_m3134_wo_DEX" "DNase_NakedK562" "DNase_NakedMcf7" )
  inList=( "DNase_UW_K562_RAW" )

  # Input Loop
  for inName in $inList
  do

    # All Strands Execution
    inObs=$inputLocation$inName"_6_obsDict.p"
    inExp=$inputLocation$inName"_6_expDict.p"
    normFileName=$normLocation$inName"_norm.txt"
    ./createKmerTable_pipeline.zsh $inObs $inExp $normFileName $outputLocation

    # Forward Strand Execution
    inObs=$inputLocation$inName"_6_obsDictF.p"
    inExp=$inputLocation$inName"_6_expDictF.p"
    normFileName=$normLocation$inName"_norm.txt"
    ./createKmerTable_pipeline.zsh $inObs $inExp $normFileName $outputLocation

    # Reverse Strand Execution
    inObs=$inputLocation$inName"_6_obsDictR.p"
    inExp=$inputLocation$inName"_6_expDictR.p"
    normFileName=$normLocation$inName"_norm.txt"
    ./createKmerTable_pipeline.zsh $inObs $inExp $normFileName $outputLocation

  done
done



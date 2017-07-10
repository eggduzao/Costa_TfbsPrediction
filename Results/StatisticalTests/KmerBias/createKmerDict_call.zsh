#!/bin/zsh

# Bam File Parameters
dloc="/hpcwork/izkf/projects/TfbsPrediction/Data/"
#dnaseList=( "DNase/H1hesc/" "DNaseUW/H7hesc/" "DNase/HeLaS3/" "DNase/HepG2/" "DNase/Huvec/" "DNase/K562/" "DNase/Mcf7/"
#            "DNaseUW/HepG2/" "DNaseUW/Huvec/" "DNaseUW/K562/" "DNaseUW/LnCaP/" "DNaseUW/m3134_with_DEX/" "DNaseUW/m3134_wo_DEX/" "DNase/NakedK562/" "DNase/NakedMcf7/" )
dnaseList=( "DNase_UW/K562_RAW/" )

# Bam File Loop
for dnase in $dnaseList
do

  # Output Type Parameters
  typeList=( "HS" "All" )
  labelList=( "CIH" "CIA" )
  allTagsList=( "N" "Y" )

  # Output Type Loop
  for i in {1..$#typeList}
  do

    # Parameters
    k_nb="6"
    allTagsFg=$allTagsList[$i]
    bamFileName=$dloc$dnase"DNase.bam"
    hsFileName=$dloc$dnase"DNasePeaks.bed"
    if [[ $dnase == "DNase_UW/m3134_with_DEX/" || $dnase == "DNase_UW/m3134_wo_DEX/" ]]; then
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/MM9/mm9.chrom.sizes"
    else
      fastaFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.fa"
      csFileName="/hpcwork/izkf/projects/TfbsPrediction/Data/HG19/hg19.chrom.sizes.filtered"
    fi
    outputLocation="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/KmerDict/"$typeList[$i]"/"

    # Execution
    bsub -J $labelList[$i] -o $labelList[$i]"_out.txt" -e $labelList[$i]"_err.txt" -W 24:00 -M 20000 -S 100 -P izkf -R "select[hpcwork]" ./createKmerDict_pipeline.zsh $k_nb $allTagsFg $bamFileName $hsFileName $fastaFileName $csFileName $outputLocation

  done
done



#!/bin/zsh

# Parameters
duLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/"
uwLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase_UW/"
outLoc="/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/StatisticalTests/KmerBias/NormFactor/"        
#inList=( $duLoc"H1hesc" $duLoc"HeLaS3" $duLoc"HepG2" $duLoc"Huvec" $duLoc"K562"
#         $uwLoc"HepG2" $uwLoc"Huvec" $uwLoc"K562" $duLoc"Mcf7" $uwLoc"LnCaP" $uwLoc"m3134_with_DEX" $uwLoc"m3134_wo_DEX" $uwLoc"H7hesc" )
#outList=( $outLoc"DNase_H1hesc" $outLoc"DNase_HeLaS3" $outLoc"DNase_HepG2" $outLoc"DNase_Huvec" $outLoc"DNase_K562"
#          $outLoc"DNaseUW_HepG2" $outLoc"DNaseUW_Huvec" $outLoc"DNaseUW_K562" $outLoc"DNase_Mcf7" $outLoc"DNaseUW_LnCaP" $outLoc"DNaseUW_m3134_with_DEX" $outLoc"DNaseUW_m3134_wo_DEX" $outLoc"DNaseUW_H7hesc" )

inList=( $uwLoc"K562_RAW" )
outList=( $outLoc"DNase_UW_K562_RAW" )

# Execution
for i in {1..$#inList}
do
  bamFileName=$inList[$i]"/DNase.bam"
  hsFileName=$inList[$i]"/DNasePeaks.bed"
  outputfileName=$outList[$i]"_norm.txt"
  bsub -J "ENF" -o "ENF_out.txt" -e "ENF_err.txt" -W 24:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./evaluateNormFactor_pipeline.zsh $bamFileName $hsFileName $outputfileName
done



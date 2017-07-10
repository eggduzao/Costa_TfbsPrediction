#!/bin/zsh

###############################
# IMR90
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/IMR90/"
inList=( "naked_dnase" )

# Execution
for inFile in $inList
do
  #bsub -J "S2B_IMR" -o "S2B_IMR_out.txt" -e "S2B_IMR_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# LnCaP
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/LnCaP/"
inList=( "AR_1nmR" "AR_10nmR" "AR_ethl" "AR_R1881" )

# Execution
for inFile in $inList
do
  #bsub -J "S2B_CAP" -o "S2B_CAP_out.txt" -e "S2B_CAP_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# m3134
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
inList=( "GR_withDEX" "GR_woDEX" "DNase_withDEX" "DNase_woDEX" )

# Execution
for inFile in $inList
do
  #bsub -J "S2B_M" -o "S2B_M_out.txt" -e "S2B_M_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# Mcf7
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Mcf7/"
inList=( "ER_0min" "ER_2min" "ER_5min" "ER_10min" "ER_40min" "ER_160min" )

# Execution
for inFile in $inList
do
  #bsub -J "S2B_MCF" -o "S2B_MCF_out.txt" -e "S2B_MCF_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# Saos2
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Saos2/"
inList=( "p53" )

# Execution
for inFile in $inList
do
  #bsub -J "S2B_SAO" -o "S2B_SAO_out.txt" -e "S2B_SAO_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# NakedK562
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedK562_2/"
inList=( "NakedK562" )

# Execution
for inFile in $inList
do
  bsub -J "S2B_NK5" -o "S2B_NK5_out.txt" -e "S2B_NK5_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done

###############################
# NakedMcf7
###############################

# Input
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedMcf7_2/"
inList=( "NakedMcf7" )

# Execution
for inFile in $inList
do
  bsub -J "S2B_NMC" -o "S2B_NMC_out.txt" -e "S2B_NMC_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./samToBam_pipeline.zsh  $bamLoc $inFile
done



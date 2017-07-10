#!/bin/zsh

###############################
# DNase-seq
###############################

# IMR90
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/IMR90/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/IMR90/"
inList=( "naked_dnase" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_D_IMR" -o "MACS_D_IMR_out.txt" -e "MACS_D_IMR_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsDNase_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done

# M3134
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
inList=( "DNase_withDEX" "DNase_woDEX" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_D_M" -o "MACS_D_M_out.txt" -e "MACS_D_M_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsDNase_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "mm"
done

# NakedK562
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/NakedK562/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/NakedK562/"
inList=( "DNase" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  bsub -J "MACS_D_NK5" -o "MACS_D_NK5_out.txt" -e "MACS_D_NK5_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsDNase_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done

# NakedMcf7
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/NakedMcf7/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/DNase/NakedMcf7/"
inList=( "DNase" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  bsub -J "MACS_D_NMC" -o "MACS_D_NMC_out.txt" -e "MACS_D_NMC_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsDNase_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done

###############################
# ChIP-seq
###############################

# LnCaP
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/LnCaP/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/LnCaP/"
inList=( "AR_1nmR" "AR_10nmR" "AR_ethl" "AR_R1881" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_C_CAP" -o "MACS_C_CAP_out.txt" -e "MACS_C_CAP_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsChIP_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done

# M3134
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
inList=( "GR_withDEX" "GR_woDEX" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_C_M" -o "MACS_C_M_out.txt" -e "MACS_C_M_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsChIP_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "mm"
done

# Mcf7
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Mcf7/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Mcf7/"
inList=( "ER_0min" "ER_2min" "ER_5min" "ER_10min" "ER_40min" "ER_160min" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_C_MCF" -o "MACS_C_MCF_out.txt" -e "MACS_C_MCF_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsChIP_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done

# Saos2
bamLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Saos2/"
peakLoc="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Saos2/"
inList=( "p53" )
for inFile in $inList
do
  mkdir -p $peakLoc$inFile"/"
  #bsub -J "MACS_C_SAO" -o "MACS_C_SAO_out.txt" -e "MACS_C_SAO_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./macsChIP_pipeline.zsh $bamLoc$inFile".bam" $inFile $peakLoc$inFile"/" "hs"
done



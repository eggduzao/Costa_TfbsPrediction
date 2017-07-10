#!/bin/zsh

###########################
# IMR90
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/IMR90/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/IMR90/"
indexName="hg19"
inList=( $il"SRR769953.fastq,"$il"SRR769954.fastq" )
outList=( $ol"naked_dnase.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_IMR" -o "BT_IMR_out.txt" -e "BT_IMR_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done


###########################
# LnCaP
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/LnCaP/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/LnCaP/"
indexName="hg19"
inList=( $il"LnCaP_1nmR_AR_GSM353642_SRR058778.fastq" 
         $il"LnCaP_10nmR_AR_GSM353641_SRR058777.fastq"
         $il"LNCaP_ethl_AR_jy9_GSM353643_SRR058779.fastq,"$il"LNCaP_ethl_AR_jy9_GSM353643_SRR058780.fastq,"$il"LNCaP_ethl_AR_jy9_GSM353643_SRR058781.fastq"
         $il"LNCaP_R1881_AR_jy10_GSM353644_SRR058757.fastq,"$il"LNCaP_R1881_AR_jy10_GSM353644_SRR058782.fastq,"$il"LNCaP_R1881_AR_jy10_GSM353644_SRR058783.fastq,"$il"LNCaP_R1881_AR_jy10_GSM353644_SRR058784.fastq,"$il"LNCaP_R1881_AR_jy10_GSM353644_SRR058785.fastq,"$il"LNCaP_R1881_AR_jy10_GSM353644_SRR058786.fastq" )
outList=( $ol"AR_1nmR.sam" $ol"AR_10nmR.sam" $ol"AR_ethl.sam" $ol"AR_R1881.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_CAP" -o "BT_CAP_out.txt" -e "BT_CAP_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done


###########################
# m3134
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/m3134/"
#indexName="mm9"
inList=( $il"ChIP_GR_withDEX_rep3_SRR088669.fastq,"$il"ChIP_GR_withDEX_rep3_SRR088670.fastq,"$il"ChIP_GR_withDEX_rep4_SRR088671.fastq,"$il"ChIP_GR_withDEX_rep4_SRR088672.fastq"
         $il"ChIP_GR_woDEX_rep3_SRR088661.fastq,"$il"ChIP_GR_woDEX_rep3_SRR088662.fastq,"$il"ChIP_GR_woDEX_rep4_SRR088663.fastq,"$il"ChIP_GR_woDEX_rep4_SRR088664.fastq"
         $il"DNase_withDEX_rep1_SRR084757.fastq,"$il"DNase_withDEX_rep1_SRR084758.fastq,"$il"DNase_withDEX_rep1_SRR084760.fastq,"$il"DNase_withDEX_rep1_SRR084761.fastq,"$il"DNase_withDEX_rep2_SRR088616.fastq,"$il"DNase_withDEX_rep2_SRR088617.fastq,"$il"DNase_withDEX_rep2_SRR088618.fastq"
         $il"DNase_woDEX_rep1_SRR084754.fastq,"$il"DNase_woDEX_rep1_SRR084755.fastq,"$il"DNase_woDEX_rep2_SRR089294.fastq,"$il"DNase_woDEX_rep2_SRR089295.fastq" )
outList=( $ol"GR_withDEX.sam" $ol"GR_woDEX.sam" $ol"DNase_withDEX.sam" $ol"DNase_woDEX.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_M" -o "BT_M_out.txt" -e "BT_M_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done


###########################
# Mcf7
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Mcf7/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Mcf7/"
indexName="hg19"
inList=( $il"ER_0min_1_GSM1325246_SRR1167674.fastq,"$il"ER_0min_1_GSM1325246_SRR1167675.fastq"
         $il"ER_2min_1_GSM1325247_SRR1167676.fastq"
         $il"ER_5min_1_GSM1325248_SRR1167677.fastq"
         $il"ER_10min_1_GSM1325249_SRR1167678.fastq,"$il"ER_10min_1_GSM1325249_SRR1167679.fastq"
         $il"ER_40min_1_GSM1325250_SRR1167680.fastq,"$il"ER_40min_1_GSM1325250_SRR1167681.fastq"
         $il"ER_160min_1_GSM1325251_SRR1167682.fastq" )
outList=( $ol"ER_0min.sam" $ol"ER_2min.sam" $ol"ER_5min.sam" $ol"ER_10min.sam" 
          $ol"ER_40min.sam" $ol"ER_160min.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_MCF" -o "BT_MCF_out.txt" -e "BT_MCF_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done


###########################
# Saos2
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Saos2/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/Saos2/"
indexName="hg19"
inList=( $il"p53_ChIPSeq_replicate1_GSM501691_SRR036613.fastq,"$il"p53_ChIPSeq_replicate1_GSM501691_SRR036614.fastq,"$il"p53_ChIPSeq_replicate2_GSM501692_SRR036615.fastq" )
outList=( $ol"p53.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_SAO" -o "BT_SAO_out.txt" -e "BT_SAO_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done

###########################
# NakedK562
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedK562/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedK562/"
indexName="hg19"
inList=( $il"SRR1565781.fastq" )
outList=( $ol"NakedK562.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  bsub -J "BT_NK5" -o "BT_NK5_out.txt" -e "BT_NK5_err.txt" -W 24:00 -M 24000 -S 100 -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done
# -P izkf

###########################
# NakedMcf7
###########################

# Input
il="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedMcf7/"
ol="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/NakedMcf7/"
indexName="hg19"
inList=( $il"SRR1565782.fastq" )
outList=( $ol"NakedMcf7.sam" )

# Running bowtie2
for i in {1..$#inList}
do
  #bsub -J "BT_NMC" -o "BT_NMC_out.txt" -e "BT_NMC_err.txt" -W 24:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./readMapping_pipeline.zsh  $indexName $inList[$i] $outList[$i]
done





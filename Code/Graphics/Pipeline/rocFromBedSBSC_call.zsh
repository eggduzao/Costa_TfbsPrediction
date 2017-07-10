#!/bin/zsh

# Cell Line Parameters
#cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" "LnCaP" "Mcf7" "Saos2" "m3134_with_DEX" "m3134_wo_DEX" )
cellList=( "K562" )

# Cell Loop
for cell in $cellList
do

  # Parameters
  if [[ $cell == "m3134_with_DEX" || $cell == "m3134_wo_DEX" ]]; then 
    il1="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/m3134_Evidence/"
  else 
    il1="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"
  fi
  il2="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/"
  il3="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/"
  outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC_NM3/"$cell"/"
  mkdir -p $outputLocation

  # Cell specific parameters
  if [[ $cell == "H1hesc" ]]; then 
    factorList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF" "EGR1" "FOSL1" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NRF1" "P300" "POU5F1" "RAD21" "REST" "RFX5" "RXRA" "SIX5" "SP1" "SP2" "SP4" "SRF" "TBP" "TCF12" "USF1" "USF2" "YY1" "ZNF143" )
    labelList="PWM,TC_DU,FOS_DU,HMMD_DU,HMMD_DU_BIAS,Boyle_DU,Neph_DU,Cuellar_DU,Pique_DU,HMMD_DU_NAKED,DNase2TF,FLR,PIQ,Wellington"

    typeList="SC,SC,SC,SB,SB,SB,SB,SC,SB,SB,SB,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"HMM_DNase_DU_ModelH1hesc_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelH1hesc_5.bed,"$il2"Boyle_DU_5.bed,"$il2"Neph_DU_5.bed,"$il2"Cuellar_DU.bed,"$il2"Centipede_DU.bed,"$il2"HMM_DNase_DU_NAKED_ModelH1hesc_5.bed,"$il2"Dnase2Tf_DU_0.bed,"$il2"FLR_DU2.bed,"$il2"PIQ_DU.bed,"$il2"Wellington_DU_0.bed"

  elif [[ $cell == "HeLaS3" ]]; then 
    factorList=( "BRCA1" "CEBPB" "CTCF" "E2F4" "E2F6" "ELK1" "FOS" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NFYA" "NFYB" "NRF1" "P300" "RAD21" "REST" "STAT1" "TBP" "USF2" "ZNF143" )
    labelList="PWM,TC_DU,FOS_DU,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"HMM_DNase_DU_ModelHeLaS3_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelHeLaS3_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelHeLaS3_5.bed"

  elif [[ $cell == "HepG2" ]]; then 
    factorList=( "ARID3A" "ATF3" "BHLHE40" "BRCA1" "CEBPB" "CTCF" "ELF1" "GABP" "JUN" "JUND" "MAFF" "MAFK" "MAX" "MYC" "NRF1" "P300" "RAD21" "REST" "RXRA" "SP1" "SP2" "SRF" "TBP" "USF1" "USF2" "YY1" )
    labelList="PWM,TC_DU,TC_UW,FOS_DU,FOS_UW,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED"
    typeList="SC,SC,SC,SC,SC,SB,SB,SB,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_DU_ModelHepG2_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelHepG2_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelHepG2_5.bed,"$il2"HMM_DNase_UW_ModelHepG2_5.bed,"$il2"HMM_DNase_UW_BIAS_ModelHepG2_5.bed,"$il2"HMM_DNase_UW_NAKED_ModelHepG2_5.bed"

  elif [[ $cell == "Huvec" ]]; then 
    factorList=( "CTCF" "FOS" "GATA2" "JUN" "MAX" "MYC" )
    labelList="PWM,TC_DU,TC_UW,FOS_DU,FOS_UW,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED"
    typeList="SC,SC,SC,SC,SC,SB,SB,SB,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_DU_ModelHuvec_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelHuvec_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelHuvec_5.bed,"$il2"HMM_DNase_UW_ModelHuvec_5.bed,"$il2"HMM_DNase_UW_BIAS_ModelHuvec_5.bed,"$il2"HMM_DNase_UW_NAKED_ModelHuvec_5.bed"

  #elif [[ $cell == "K562" ]]; then 
  #  factorList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" "EJUND" "ELF1" "ELK1" "ETS1" "FOS" "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" "MAFF" "MAFK" "MAX" "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" "RFX5" "SIX5" "SMC3" "SP1" "SP2" "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" "TBP" "THAP1" "TR4" "USF1" "USF2" "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
    #labelList="PWM,TC_DU,TC_UW,FOS_DU,FOS_UW,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED,Boyle_DU,Neph_DU,Cuellar_DU,Pique_DU,DNase2TF,FLR,PIQ,Wellington"
    #typeList="SC,SC,SC,SC,SC,SB,SB,SB,SB,SB,SB,SB,SB,SC,SB,SB,SB,SB,SB"
    #bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_DU_ModelK562_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelK562_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelK562_5.bed,"$il2"HMM_DNase_UW_ModelK562_5.bed,"$il2"HMM_DNase_UW_BIAS_ModelK562_5.bed,"$il2"HMM_DNase_UW_NAKED_ModelK562_5.bed,"$il2"Boyle_DU_5.bed,"$il2"Neph_DU_5.bed,"$il2"Cuellar_DU.bed,"$il2"Centipede_DU.bed,"$il2"Dnase2Tf_DU_0.bed,"$il2"FLR_DU2.bed,"$il2"PIQ_DU.bed,"$il2"Wellington_DU_0.bed"
  elif [[ $cell == "K562" ]]; then 
    factorList=( "SIX5" )
    labelList="PWM,TC_DU,FOS_DU,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED,PIQ"
    typeList="SC,SC,SC,SB,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TC_DU.bed,"$il2"FS_DU.bed,"$il2"HINT_D_DU.bed,"$il2"HINT-BC_D_DU.bed,"$il2"HINT-BCN_D_DU.bed,"$il2"PIQ_99.bed"

  elif [[ $cell == "LnCaP" ]]; then 
    factorList=( "AR_10nmR"  "AR_1nmR"  "AR_R1881"  "AR_ethl" )
    labelList="PWM,TC_UW,FOS_UW,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_UW.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_UW_ModelLnCaP_5.bed,"$il2"HMM_DNase_UW_BIAS_ModelLnCaP_5.bed,"$il2"HMM_DNase_UW_NAKED_ModelLnCaP_5.bed"
    
  elif [[ $cell == "Mcf7" ]]; then 
    factorList=( "ER_0min" "ER_160min" "ER_40min" "ER_10min" "ER_2min" "ER_5min" )
    labelList="PWM,TC_DU,FOS_DU,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"HMM_DNase_DU_ModelMcf7_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelMcf7_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelMcf7_5.bed"
    
  elif [[ $cell == "Saos2" ]]; then 
    factorList=( "P53" )
    labelList="PWM,TC_DU,FOS_DU,HMMD_DU,HMMD_DU_BIAS,HMMD_DU_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"HMM_DNase_DU_ModelSaos2_5.bed,"$il2"HMM_DNase_DU_BIAS_ModelSaos2_5.bed,"$il2"HMM_DNase_DU_NAKED_ModelSaos2_5.bed"
   
  elif [[ $cell == "m3134_with_DEX" ]]; then 
    factorList=( "GR_withDEX" "GR_woDEX" )
    labelList="PWM,TC_UW,FOS_UW,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_UW.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_UW_Modelm3134_5.bed,"$il2"HMM_DNase_UW_BIAS_Modelm3134_5.bed,"$il2"HMM_DNase_UW_NAKED_Modelm3134_5.bed"
    
  elif [[ $cell == "m3134_wo_DEX" ]]; then 
    factorList=( "GR_withDEX" "GR_woDEX" )
    labelList="PWM,TC_UW,FOS_UW,HMMD_UW,HMMD_UW_BIAS,HMMD_UW_NAKED"
    typeList="SC,SC,SC,SB,SB,SB"
    bedList=$il1"fdr_4.bed,"$il2"TagCount_UW.bed,"$il2"FOS_UW.bed,"$il2"HMM_DNase_UW_Modelm3134_5.bed,"$il2"HMM_DNase_UW_BIAS_Modelm3134_5.bed,"$il2"HMM_DNase_UW_NAKED_Modelm3134_5.bed"
    
  fi

  # Factor Loop
  for factor in $factorList
  do
    bsub -J $cell"_"$factor"_ROCS" -o $cell"_"$factor"_ROCS_out.txt" -e $cell"_"$factor"_ROCS_err.txt" -W 48:00 -M 24000 -S 100 -P izkf -R "select[hpcwork]" ./rocFromBedSBSC_pipeline.zsh $factor $typeList $labelList $bedList $outputLocation
  done
done



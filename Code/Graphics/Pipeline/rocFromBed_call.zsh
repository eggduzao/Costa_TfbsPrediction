#!/bin/zsh

# Global Parameters
outExt="png"
allNeg="y"

# Cell Line Parameters
cellList=( "H1hesc" "HeLaS3" "HepG2" "Huvec" "K562" )

# Cell Loop
for cell in $cellList
do

    # Evidence Parameters
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Parameters
        il1="/hpcwork/izkf/projects/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"
        il2="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/Results/"$cell"/"$ev"/"
        outputLocation="/hpcwork/izkf/projects/TfbsPrediction/Results/Footprints/ROC/"$cell"/"$ev"/"
        mkdir -p $outputLocation

        # Cell specific parameters
        if [[ $cell == "H1hesc" ]]; then 
            factorList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF" "EGR1" "FOSL1" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NRF1" "P300"
                         "POU5F1" "RAD21" "REST" "RFX5" "RXRA" "SIX5" "SP1" "SP2" "SP4" "SRF" "TBP" "TCF12" "USF1" "USF2" "YY1" "ZNF143" )
            labelList="PWM,DNase_DU,DNaseS_DU,D13_DU,D13S_DU,TagCount_DU,FOS_DU,Boyle,Neph_5,Cuellar_D13,Cuellar_D139,Pique_ces,DH-HMM_13_DU(H),DH-HMM_13_DU(K),DH-HMM_13_DU_BIAS(H),DH-HMM_139_DU(H),DH-HMM_139_DU(K),H-HMM_13(H),H-HMM_13(K),H-HMM_139(H),H-HMM_139(K)"
            bedList=$il1$ev".bed,"$il2"DNase_DU.bed,"$il2"DNase_DU_WithScore.bed,"$il2"D13_DU.bed,"$il2"D13_DU_WithScore.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"Boyle_5.bed,"$il2"Neph_5.bed,"$il2"Cuellar5_D13.bed,"$il2"Cuellar5_D139.bed,"$il2"Pique_ces.bed,"$il2"HD_D13_ModelH1hesc_5.bed,"$il2"HD_D13_ModelK562_5.bed,"$il2"HMM_DU_BIAS_D13_ModelH1hesc_5.bed,"$il2"HD_D139_ModelH1hesc_5.bed,"$il2"HD_D139_ModelK562_5.bed,"$il2"HMMH_D13_ModelH1hesc_0.bed,"$il2"HMMH_D13_ModelK562_0.bed,"$il2"HMMH_D139_ModelH1hesc_0.bed,"$il2"HMMH_D139_ModelK562_0.bed"
        elif [[ $cell == "HeLaS3" ]]; then 
            factorList=( "BRCA1" "CEBPB" "CTCF" "E2F4" "E2F6" "ELK1" "FOS" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NFYA" "NFYB"
                         "NRF1" "P300" "RAD21" "REST" "STAT1" "TBP" "USF2" "ZNF143" )
            labelList="PWM,DNase_DU,DNaseS_DU,D13_DU,D13S_DU,TagCount_DU,FOS_DU,DH-HMM_13_DU(H1),DH-HMM_13_DU(HL),DH-HMM_13_DU(HP),DH-HMM_13_DU(K),DH-HMM_13_DU_BIAS(HL)"
            bedList=$il1$ev".bed,"$il2"DNase_DU.bed,"$il2"DNase_DU_WithScore.bed,"$il2"D13_DU.bed,"$il2"D13_DU_WithScore.bed,"$il2"TagCount_DU.bed,"$il2"FOS_DU.bed,"$il2"HD_D13_ModelH1hesc_5.bed,"$il2"HD_D13_ModelHeLaS3_5.bed,"$il2"HD_D13_ModelHepG2_5.bed,"$il2"HD_D13_ModelK562_5.bed,"$il2"HMM_DU_BIAS_D13_ModelHeLaS3_5.bed"
        elif [[ $cell == "HepG2" ]]; then 
            factorList=( "ARID3A" "ATF3" "BHLHE40" "BRCA1" "CEBPB" "CTCF" "ELF1" "GABP" "JUN" "JUND" "MAFF" "MAFK" "MAX" "MYC" "NRF1"
                         "P300" "RAD21" "REST" "RXRA" "SP1" "SP2" "SRF" "TBP" "USF1" "USF2" "YY1" )
            labelList="PWM,DNase_DU,DNaseS_DU,D13_DU,D13S_DU,DNase_UW,DNaseS_UW,D13_UW,D13S_UW,TagCount_DU,TagCount_UW,FOS_DU,FOS_UW,DH-HMM_13_DU(H1),DH-HMM_13_DU(HL),DH-HMM_13_DU(HP),DH-HMM_13_DU(K),DH-HMM_13_DU_BIAS(HP),DH-HMM_13_UW(HP),DH-HMM_13_UW_BIAS(HP)"
            bedList=$il1$ev".bed,"$il2"DNase_DU.bed,"$il2"DNase_DU_WithScore.bed,"$il2"D13_DU.bed,"$il2"D13_DU_WithScore.bed,"$il2"DNase_UW.bed,"$il2"DNase_UW_WithScore.bed,"$il2"D13_UW.bed,"$il2"D13_UW_WithScore.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"HD_D13_ModelH1hesc_5.bed,"$il2"HD_D13_ModelHeLaS3_5.bed,"$il2"HD_D13_ModelHepG2_5.bed,"$il2"HD_D13_ModelK562_5.bed,"$il2"HMM_DU_BIAS_D13_ModelHepG2_5.bed,"$il2"HMM_UW_D13_ModelHepG2_5.bed,"$il2"HMM_UW_BIAS_D13_ModelHepG2_5.bed"
        elif [[ $cell == "Huvec" ]]; then 
            factorList=( "CTCF" "FOS" "GATA2" "JUN" "MAX" "MYC" )
            labelList="PWM,DNase_DU,DNaseS_DU,D13_DU,D13S_DU,DNase_UW,DNaseS_UW,D13_UW,D13S_UW,TagCount_DU,TagCount_UW,FOS_DU,FOS_UW,DH-HMM_13_DU(HU),DH-HMM_13_DU_BIAS(HU),DH-HMM_13_UW(HU),DH-HMM_13_UW_BIAS(HU)"
            bedList=$il1$ev".bed,"$il2"DNase_DU.bed,"$il2"DNase_DU_WithScore.bed,"$il2"D13_DU.bed,"$il2"D13_DU_WithScore.bed,"$il2"DNase_UW.bed,"$il2"DNase_UW_WithScore.bed,"$il2"D13_UW.bed,"$il2"D13_UW_WithScore.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"HD_D13_ModelHuvec_5.bed,"$il2"HMM_DU_BIAS_D13_ModelHuvec_5.bed,"$il2"HMM_UW_D13_ModelHuvec_5.bed,"$il2"HMM_UW_BIAS_D13_ModelHuvec_5.bed"
        else
            factorList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" 
                         "EJUND" "ELF1" "ELK1" "ETS1" "FOS" "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" "MAFF" "MAFK" "MAX" 
                         "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" "RFX5" "SIX5" "SMC3" "SP1" "SP2" 
                         "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" "TBP" "THAP1" "TR4" "USF1" "USF2" "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
            labelList="PWM,DNase_DU,DNaseS_DU,D13_DU,D13S_DU,DNase_UW,DNaseS_UW,D13_UW,D13S_UW,TagCount_DU,TagCount_UW,FOS_DU,FOS_UW,Boyle,Neph_5,Cuellar_D13,Cuellar_D139,Pique_ces,DH-HMM_13_DU(H),DH-HMM_13_DU(K),DH-HMM_13_DU_BIAS(K),DH-HMM_13_UW(K),DH-HMM_13_UW_BIAS(K),DH-HMM_139_DU(H),DH-HMM_139_DU(K),H-HMM_13(H),H-HMM_13(K),H-HMM_139(H),H-HMM_139(K)"
            bedList=$il1$ev".bed,"$il2"DNase_DU.bed,"$il2"DNase_DU_WithScore.bed,"$il2"D13_DU.bed,"$il2"D13_DU_WithScore.bed,"$il2"DNase_UW.bed,"$il2"DNase_UW_WithScore.bed,"$il2"D13_UW.bed,"$il2"D13_UW_WithScore.bed,"$il2"TagCount_DU.bed,"$il2"TagCount_UW.bed,"$il2"FOS_DU.bed,"$il2"FOS_UW.bed,"$il2"Boyle_5.bed,"$il2"Neph_5.bed,"$il2"Cuellar5_D13.bed,"$il2"Cuellar5_D139.bed,"$il2"Pique_ces.bed,"$il2"HD_D13_ModelH1hesc_5.bed,"$il2"HD_D13_ModelK562_5.bed,"$il2"HMM_DU_BIAS_D13_ModelK562_5.bed,"$il2"HMM_UW_D13_ModelK562_5.bed,"$il2"HMM_UW_BIAS_D13_ModelK562_5.bed,"$il2"HD_D139_ModelH1hesc_5.bed,"$il2"HD_D139_ModelK562_5.bed,"$il2"HMMH_D13_ModelH1hesc_0.bed,"$il2"HMMH_D13_ModelK562_0.bed,"$il2"HMMH_D139_ModelH1hesc_0.bed,"$il2"HMMH_D139_ModelK562_0.bed"

        fi

        # Factor Loop
        for factor in $factorList
        do
            bsub -J $cell"_"$factor"_ROC" -o $cell"_"$factor"_ROC_out.txt" -e $cell"_"$factor"_ROC_err.txt" -W 3:00 -M 12000 -S 100 -R "select[hpcwork]" ./rocFromBed_pipeline.zsh $allNeg $outExt $factor $labelList $bedList $outputLocation
            # -sp 10 -P izkf
        done
    done
done



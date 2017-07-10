#!/bin/zsh

# Global Parameters
normVec="y"
windowLen="10000"

# Cell Variation
#cellList=( "H1hesc" "HeLaS3" "HepG2" "K562" )
cellList=( "H1hesc" "K562" )

# Cell Loop
for cell in $cellList
do
   
    # Parameters
    pl1="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/"
    pl2="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/"$cell"/Extension/"


    # Cell specific parameters
    if [[ $cell == "H1hesc" ]]; then    
        factorList=( "ATF3" "BACH1" "BRCA1" "CEBPB" "CTCF" "EGR1" "FOSL1" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NRF1" "P300"
                     "POU5F1" "RAD21" "REST" "RFX5" "RXRA" "SIX5" "SP1" "SP2" "SP4" "SRF" "TBP" "TCF12" "USF1" "USF2" "YY1" "ZNF143" )
        predLabelList="Boyle,Neph_0,Neph_5,DH-HMM_13(H),DH-HMM_13(K),DH-HMM_139(H),DH-HMM_139(K),H-HMM_13(H),H-HMM_13(K),H-HMM_139(H),H-HMM_139(K)"
        predFileNameList=$pl2"Boyle_5.bed,"$pl2"Neph_0.bed,"$pl2"Neph_5.bed,"$pl2"HD_D13_ModelH1hesc_5.bed,"$pl2"HD_D13_ModelK562_5.bed,"$pl2"HD_D139_ModelH1hesc_5.bed,"$pl2"HD_D139_ModelK562_5.bed,"$pl2"HMMH_D13_ModelH1hesc_5.bed,"$pl2"HMMH_D13_ModelK562_5.bed,"$pl2"HMMH_D139_ModelH1hesc_5.bed,"$pl2"HMMH_D139_ModelK562_5.bed"
    elif [[ $cell == "HeLaS3" ]]; then 
        factorList=( "BRCA1" "CEBPB" "CTCF" "E2F4" "E2F6" "ELK1" "FOS" "GABP" "JUN" "JUND" "MAFK" "MAX" "MYC" "NFYA" "NFYB"
                     "NRF1" "P300" "RAD21" "REST" "STAT1" "TBP" "USF2" "ZNF143" )
        predLabelList="DH-HMM_13(H1),DH-HMM_13(HL),DH-HMM_13(HP),DH-HMM_13(K)"
        predFileNameList=$pl2"HD_D13_ModelH1hesc_5.bed,"$pl2"HD_D13_ModelHeLaS3_5.bed,"$pl2"HD_D13_ModelHepG2_5.bed,"$pl2"HD_D13_ModelK562_5.bed"
    elif [[ $cell == "HepG2" ]]; then 
        factorList=( "ARID3A" "ATF3" "BHLHE40" "BRCA1" "CEBPB" "CTCF" "ELF1" "GABP" "JUN" "JUND" "MAFF" "MAFK" "MAX" "MYC" "NRF1"
                     "P300" "RAD21" "REST" "RXRA" "SP1" "SP2" "SRF" "TBP" "USF1" "USF2" "YY1" )
        predLabelList="DH-HMM_13(H1),DH-HMM_13(HL),DH-HMM_13(HP),DH-HMM_13(K)"
        predFileNameList=$pl2"HD_D13_ModelH1hesc_5.bed,"$pl2"HD_D13_ModelHeLaS3_5.bed,"$pl2"HD_D13_ModelHepG2_5.bed,"$pl2"HD_D13_ModelK562_5.bed"
    else
        factorList=( "ARID3A" "ATF1" "ATF3" "BACH1" "BHLHE40" "CCNT2" "CEBPB" "CTCF" "CTCFL" "E2F4" "E2F6" "EFOS" "EGATA" "EGR1" "EJUNB" 
                 "EJUND" "ELF1" "ELK1" "ETS1" "FOS" "FOSL1" "GABP" "GATA1" "GATA2" "IRF1" "JUN" "JUND" "MAFF" "MAFK" "MAX" 
                 "MEF2A" "MYC" "NFE2" "NFYA" "NFYB" "NR2F2" "NRF1" "PU1" "RAD21" "REST" "RFX5" "SIX5" "SMC3" "SP1" "SP2" 
                 "SRF" "STAT1" "STAT2" "STAT5A" "TAL1" "TBP" "THAP1" "TR4" "USF1" "USF2" "YY1" "ZBTB7A" "ZBTB33" "ZNF143" "ZNF263" )
        predLabelList="Boyle,Neph_0,Neph_5,DH-HMM_13(H),DH-HMM_13(K),DH-HMM_139(H),DH-HMM_139(K),H-HMM_13(H),H-HMM_13(K),H-HMM_139(H),H-HMM_139(K)"
        predFileNameList=$pl2"Boyle_5.bed,"$pl2"Neph_0.bed,"$pl2"Neph_5.bed,"$pl2"HD_D13_ModelH1hesc_5.bed,"$pl2"HD_D13_ModelK562_5.bed,"$pl2"HD_D139_ModelH1hesc_5.bed,"$pl2"HD_D139_ModelK562_5.bed,"$pl2"HMMH_D13_ModelH1hesc_5.bed,"$pl2"HMMH_D13_ModelK562_5.bed,"$pl2"HMMH_D139_ModelH1hesc_5.bed,"$pl2"HMMH_D139_ModelK562_5.bed"
    fi

    # Evidence Variation
    evList=( "fdr_4" )

    # Evidence Loop
    for ev in $evList
    do

        # Parameters
        mpbsFileName="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/MPBSAWG/"$cell"_Evidence/"$ev".bed"
        outputLocation="/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Histogram/"$cell"/"$ev"/"
        mkdir -p $outputLocation
        
        # Factor Loop
        for factor in $factorList
        do
            bsub -J $cell"_"$ev"_"$factor"_HIS" -o $cell"_"$ev"_"$factor"_HIS_out.txt" -e $cell"_"$ev"_"$factor"_HIS_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./predictionHistogram_pipeline.zsh $normVec $windowLen $factor $mpbsFileName $predLabelList $predFileNameList $outputLocation
        done

        # All factors execution
        bsub -J $cell"_"$ev"_all_HIS" -o $cell"_"$ev"_all_HIS_out.txt" -e $cell"_"$ev"_all_HIS_err.txt" -W 100:00 -M 12000 -S 100 -P izkf -R "select[hpcwork]" ./predictionHistogram_pipeline.zsh $normVec $windowLen "all" $mpbsFileName $predLabelList $predFileNameList $outputLocation

    done
done



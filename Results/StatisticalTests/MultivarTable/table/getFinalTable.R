
# Reading tables
inLoc = "./"
inList = c("DU_H1hesc", "DU_HeLaS3", "DU_HepG2", "DU_Huvec", 
           "DU_K562", "DU_Mcf7", "UW_denovo", "UW_HepG2", "UW_Huvec", 
           "UW_K562", "UW_LnCaP", "UW_m3134_with_DEX")
data = c()
for(ifn in inList){
 currdata = read.table(paste(inLoc,ifn,".txt",sep=""), sep='\t', header=T)
 data = rbind(data,currdata)
 colnames(data) = colnames(currdata)
}

# Selecting columns
colSel = c(
"LAB", "CELL", "FACTOR", 
"PWM_REPOSITORY", "PWM_ID", "CHIPSEQ_LAB", "CHIPSEQ_ID", 
"NUMBER_MOTIFS", "NUMBER_PEAKS", "NUMBER_PEAKS_with_MOTIFS", "PERC_PEAKS_MOTIFS",
"BIAS_PEARSON", "MOTIF_CG_CONTENT",
"MINSUM_PROT_AVG_FP_CORRECTED", 
"BinDNase_80_AUC_1", "BinDNase_85_AUC_1", "BinDNase_90_AUC_1", "BinDNase_95_AUC_1", "BinDNase_99_AUC_1", "BinDNase_rank_AUC_1", "Boyle_AUC_1", "Centipede_80_AUC_1", "Centipede_85_AUC_1", "Centipede_90_AUC_1", "Centipede_95_AUC_1", "Centipede_99_AUC_1", "Cuellar_80_AUC_1", "Cuellar_85_AUC_1", "Cuellar_90_AUC_1", "Cuellar_95_AUC_1", "Cuellar_99_AUC_1", "DNase2TF_TC_AUC_1", "DNase2TF_rank_AUC_1", "FLR_80_AUC_1", "FLR_85_AUC_1", "FLR_90_AUC_1", "FLR_95_AUC_1", "FLR_99_AUC_1", "FS_AUC_1", "HINTBCN_AUC_1", "HINTBC_AUC_1", "HINT_AUC_1", "Neph_AUC_1", "PIQ_80_AUC_1", "PIQ_85_AUC_1", "PIQ_90_AUC_1", "PIQ_95_AUC_1", "PIQ_99_AUC_1", "PWM_AUC_1", "Protection_AUC_1", "TC_AUC_1", "Wellington_TC_AUC_1", "Wellington_rank_AUC_1", "TCH_DU_AUC_1", "HINT_H13_DU_AUC_1", "HINT_D13_DU_AUC_1", "HINT_BC_D13_DU_AUC_1", "ATAC_TC_50_AUC_1", "scATAC_TC_50_AUC_1", "ATAC_HINT_twoside_1_AUC_1", "sc_ATAC_HINT_twoside_1_AUC_1", "ATAC_HINT_twoside_1_SHIFT_AUC_1", "ATAC_HINT_twoside_1_SHIFT_BC_AUC_1", "sc_ATAC_HINT_twoside_1_SHIFT_AUC_1", "sc_ATAC_HINT_twoside_1_SHIFT_BC_AUC_1", 
"PWM_AUC_10", "TC_AUC_10", "FOS_AUC_10", "HINT_AUC_10", "HINT_BC_AUC_10", "HINT_BCN_AUC_10", "Boyle_AUC_10", "Neph_AUC_10", "Cuellar_AUC_10", "Centipede_AUC_10", "DNase2TF_AUC_10", "FLR_AUC_10", "PIQ_AUC_10", "Wellington_AUC_10", "BinDNase_80_AUC_10", "BinDNase_85_AUC_10", "BinDNase_90_AUC_10", "BinDNase_95_AUC_10", "BinDNase_99_AUC_10", "BinDNase_rank_AUC_10", "Centipede_80_AUC_10", "Centipede_85_AUC_10", "Centipede_95_AUC_10", "Centipede_99_AUC_10", "Cuellar_80_AUC_10", "Cuellar_85_AUC_10", "Cuellar_95_AUC_10", "Cuellar_99_AUC_10", "DNase2TF_rank_AUC_10", "FLR_80_AUC_10", "FLR_85_AUC_10", "FLR_95_AUC_10", "FLR_99_AUC_10", "PIQ_80_AUC_10", "PIQ_85_AUC_10", "PIQ_95_AUC_10", "PIQ_99_AUC_10", "Protection_AUC_10", "Wellington_rank_AUC_10", "TCH_DU_AUC_10", "HINT_H13_DU_AUC_10", "HINT_D13_DU_AUC_10", "HINT_BC_D13_DU_AUC_10", "ATAC_TC_50_AUC_10", "scATAC_TC_50_AUC_10",  "ATAC_HINT_twoside_1_AUC_10", "sc_ATAC_HINT_twoside_1_AUC_10", "ATAC_HINT_twoside_1_SHIFT_AUC_10", "ATAC_HINT_twoside_1_SHIFT_BC_AUC_10", "sc_ATAC_HINT_twoside_1_SHIFT_AUC_10", "sc_ATAC_HINT_twoside_1_SHIFT_BC_AUC_10", 
"PWM_AUC_ALL", "TC_AUC_ALL", "FOS_AUC_ALL", "HINT_AUC_ALL", "HINT_BC_AUC_ALL", "HINT_BCN_AUC_ALL", "Boyle_AUC_ALL", "Neph_AUC_ALL", "Cuellar_AUC_ALL", "Centipede_AUC_ALL", "DNase2TF_AUC_ALL", "FLR_AUC_ALL", "PIQ_AUC_ALL", "Wellington_AUC_ALL", "BinDNase_80_AUC_ALL", "BinDNase_85_AUC_ALL", "BinDNase_90_AUC_ALL", "BinDNase_95_AUC_ALL", "BinDNase_99_AUC_ALL", "BinDNase_rank_AUC_ALL", "Centipede_80_AUC_ALL", "Centipede_85_AUC_ALL", "Centipede_95_AUC_ALL", "Centipede_99_AUC_ALL", "Cuellar_80_AUC_ALL", "Cuellar_85_AUC_ALL", "Cuellar_95_AUC_ALL", "Cuellar_99_AUC_ALL", "DNase2TF_rank_AUC_ALL", "FLR_80_AUC_ALL", "FLR_85_AUC_ALL", "FLR_95_AUC_ALL", "FLR_99_AUC_ALL", "PIQ_80_AUC_ALL", "PIQ_85_AUC_ALL", "PIQ_95_AUC_ALL", "PIQ_99_AUC_ALL", "Protection_AUC_ALL", "Wellington_rank_AUC_ALL", "TCH_DU_AUC_ALL", "HINT_H13_DU_AUC_ALL", "HINT_D13_DU_AUC_ALL", "HINT_BC_D13_DU_AUC_ALL", "ATAC_TC_50_AUC_ALL", "scATAC_TC_50_AUC_ALL", "ATAC_HINT_twoside_1_AUC_ALL", "sc_ATAC_HINT_twoside_1_AUC_ALL", "ATAC_HINT_twoside_1_SHIFT_AUC_ALL", "ATAC_HINT_twoside_1_SHIFT_BC_AUC_ALL", "sc_ATAC_HINT_twoside_1_SHIFT_AUC_ALL",  "sc_ATAC_HINT_twoside_1_SHIFT_BC_AUC_ALL", 
"PWM_AUPR_ALL", "TC_AUPR_ALL", "FOS_AUPR_ALL", "HINT_AUPR_ALL", "HINT_BC_AUPR_ALL", "HINT_BCN_AUPR_ALL",   "Boyle_AUPR_ALL", "Neph_AUPR_ALL", "Cuellar_AUPR_ALL", "Centipede_AUPR_ALL", "DNase2TF_AUPR_ALL", "FLR_AUPR_ALL", "PIQ_AUPR_ALL", "Wellington_AUPR_ALL", "BinDNase_80_AUPR_ALL", "BinDNase_85_AUPR_ALL", "BinDNase_90_AUPR_ALL", "BinDNase_95_AUPR_ALL", "BinDNase_99_AUPR_ALL", "BinDNase_rank_AUPR_ALL", "Centipede_80_AUPR_ALL", "Centipede_85_AUPR_ALL", "Centipede_95_AUPR_ALL", "Centipede_99_AUPR_ALL", "Cuellar_80_AUPR_ALL", "Cuellar_85_AUPR_ALL", "Cuellar_95_AUPR_ALL", "Cuellar_99_AUPR_ALL", "DNase2TF_rank_AUPR_ALL", "FLR_80_AUPR_ALL", "FLR_85_AUPR_ALL", "FLR_95_AUPR_ALL", "FLR_99_AUPR_ALL", "PIQ_80_AUPR_ALL", "PIQ_85_AUPR_ALL", "PIQ_95_AUPR_ALL", "PIQ_99_AUPR_ALL", "Protection_AUPR_ALL", "Wellington_rank_AUPR_ALL", "TCH_DU_AUPR_ALL", "HINT_H13_DU_AUPR_ALL", "HINT_D13_DU_AUPR_ALL", "HINT_BC_D13_DU_AUPR_ALL", "ATAC_TC_50_AUPR_ALL", "scATAC_TC_50_AUPR_ALL", "ATAC_HINT_twoside_1_AUPR_ALL", "sc_ATAC_HINT_twoside_1_AUPR_ALL", "ATAC_HINT_twoside_1_SHIFT_AUPR_ALL", "ATAC_HINT_twoside_1_SHIFT_BC_AUPR_ALL", "sc_ATAC_HINT_twoside_1_SHIFT_AUPR_ALL", "sc_ATAC_HINT_twoside_1_SHIFT_BC_AUPR_ALL")

data = data[,colSel]
outputFileName = "./final_table.csv"
write.csv(data,file=outputFileName)



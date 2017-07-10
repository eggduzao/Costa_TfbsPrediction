
# Column selection
#methodVec = c("Boyle", "Centipede_80", "Centipede_85", "Centipede_90", "Centipede_95", "Centipede_99", "Cuellar_80", "Cuellar_85", "Cuellar_90", "Cuellar_85", "Cuellar_99", "Dnase2Tf", "FLR_80", "FLR_85", "FLR_90", "FLR_95", "FLR_99", "FS", "HINTBCN", "HINTBC", "HINT", "Neph", "PIQ_80", "PIQ_85", "PIQ_90", "PIQ_95", "PIQ_99", "PWM", "TC", "Wellington")
methodVec = c("Boyle", "Centipede_90", "Cuellar_90", "Dnase2Tf", "FLR_90", "FS", "HINTBCN", "HINTBC", "HINT", "Neph", "PIQ_90", "PWM", "TC", "Wellington")
metricVec = c("flr", "fos", "tc")
#finalVec = c("ks_stat", "ks_pvalue")
finalVec = c("ks_stat")
colSel = c()
for(m in methodVec){
  for(me in metricVec){
    for(f in finalVec){
      colSel = c(colSel,paste(m,me,f,sep="_"))
    }
  }
}
colSel = c("FACTOR", "EXPR_FOLDCHANGE", "EXPR_H1hesc", "EXPR_GM12878", colSel)

# Reading and writing data
data = read.table("./multivar_table/multivar_table.txt", sep='\t', header=T)
data = data[,colSel]
outputFileName = "./final_table.csv"
write.csv(data,file=outputFileName)



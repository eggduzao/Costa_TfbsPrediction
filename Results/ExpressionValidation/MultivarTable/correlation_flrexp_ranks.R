
###########################################################################################################################################
# Correlation Function
###########################################################################################################################################

# Creating Correlation Function
createCorrelation <- function(vec1, vec2, formula, data, corrtype){
  regLine = lm(formula, data=data) # Regression line (y,x)
  corrTest = cor.test(vec1, vec2, alternative = "two.sided", method = corrtype, conf.level = 0.95) # Correlation
  pValue = corrTest$p.value
  corr = corrTest$estimate
  return(round(corr,digits = 6))
}

# Function to standardize values
standardize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

###########################################################################################################################################
# Correlation 
###########################################################################################################################################

corrtype = "spearman"
data = read.table("./multivar_table_all_methods.txt", sep=',', header=T)

name1 = "DEC_GK_FLR"
name2 = "DEC_HG_FLR"
vec1 = data[,name1]
vec2 = data[,name2]
data.sampled = data.frame(X=vec1, Y=vec2)
cor = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled, corrtype)
print(cor)

name1 = "DEC_GK_FLR"
name2 = "DEC_HK_FLR"
vec1 = data[,name1]
vec2 = data[,name2]
data.sampled = data.frame(X=vec1, Y=vec2)
cor = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled, corrtype)
print(cor)

name1 = "DEC_HK_FLR"
name2 = "DEC_HG_FLR"
vec1 = data[,name1]
vec2 = data[,name2]
data.sampled = data.frame(X=vec1, Y=vec2)
cor = createCorrelation(data.sampled[,1], data.sampled[,2], Y~X, data.sampled, corrtype)
print(cor)


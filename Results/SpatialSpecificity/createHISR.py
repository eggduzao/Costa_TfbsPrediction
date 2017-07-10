#####################################################################################
# Script to create the R script that will create all the HIS figures.
#####################################################################################

# Import
import os
import sys
import glob

# Input
threshold = sys.argv[1] # Maximum distance value to be considered. Eg. 2000 (Max is 5000 == all data)
cell = sys.argv[2] # Cell line label. Eg. H1hesc
ev = sys.argv[3] # Evidence label. Eg. fpr_4
hisFolder = sys.argv[4] # Folder to the HIS curve results. Inside this folder the results must be separated by cell line / evidence.
outHisLoc = sys.argv[5] # Output location of HIS curves.

#################################################
# Parameters
#################################################

# Cell translation
cellFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/Cell.txt"
cellTranslation = dict()
cellFile = open(cellFileName,"r")
for line in cellFile:
    ll = line.strip().split("\t")
    cellTranslation[ll[0]] = ll[1]
cellFile.close()

# TF translation
factorFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/TF.txt"
factorTranslation = dict()
factorFile = open(factorFileName,"r")
for line in factorFile:
    ll = line.strip().split("\t")
    factorTranslation[ll[0]] = ll[1]
factorFile.close()

# Method translation
methodFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/data/Results_Bioinformatics2014/LabelTranslation/HIS_Method_"+cell+".txt"
labelList = []; fencyLabelList = []
labelFile = open(methodFileName,"r")
for line in labelFile:
    ll = line.strip().split("\t")
    labelList.append(ll[0])
    fencyLabelList.append(ll[1])
labelFile.close()

#################################################
# HIS / Tex Initialization
#################################################

# Creating HIS output
currOutHisLoc = outHisLoc+cell+"/"+ev+"/"
os.system("mkdir -p "+currOutHisLoc)
outputFileName = currOutHisLoc+"script.R"
outputFile = open(outputFileName,"w")
outputTexFileName = currOutHisLoc+"script.tex"
outputTexFile = open(outputTexFileName,"w")

# Writing global parameters to R script
outputFile.write("library(lattice)\n")
outputFile.write("library(reshape)\n")
outputFile.write("w2l <- function(xx) melt(xx, measure.vars = colnames(xx))\n")
outputFile.write("\n")

# Writing global parameters to tex script
outputTexFile.write("\\documentclass[11pt,a4]{article}\n")
outputTexFile.write("\\usepackage[cm]{fullpage}\n")
outputTexFile.write("\\usepackage{graphicx}\n")
outputTexFile.write("\\usepackage{subfigure}\n")
outputTexFile.write("\\usepackage{epsfig}\n")
outputTexFile.write("\\usepackage{epstopdf}\n")
outputTexFile.write("\\begin{document}\n\n")
outputTexFile.write("\\begin{figure}[h]\n")
outputTexFile.write("\\centering\n")
labelCounter = 1
factorCounter = 0

#################################################
# HIS / Tex Print
#################################################

# Iterating on factors
for hisFileName in glob.glob(hisFolder+cell+"/"+ev+"/*_box.txt"):

    # Fetching columns
    factor = "_".join(hisFileName.split("/")[-1].split("_")[:-1])
    hisFile = open(hisFileName,"r")
    headerList = hisFile.readline().strip().split("\t")
    posList = []
    newFencyLabelList = []
    for i in range(0,len(labelList)):
        if(labelList[i] in headerList):
            posList.append(str(headerList.index(labelList[i])+1))
            newFencyLabelList.append(fencyLabelList[i])
    hisFile.close()

    # Writing R script
    outputFile.write("# "+cell+" - "+ev+" - "+factor+" \n")
    outputFile.write("postscript('"+currOutHisLoc+factor+".eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')\n") # Save to eps
    outputFile.write("par(cex.axis=0.8)\n")
    outputFile.write("bx=read.table('"+hisFileName+"',header=TRUE)\n")
    outputFile.write("bx=replace(bx, abs(bx) > "+threshold+", NA)\n")
    outputFile.write("bx2 = abs(bx[,c("+",".join(posList)+")])\n")
    outputFile.write("methodNameVec = c("+",".join(["'"+e+"'" for e in newFencyLabelList])+")\n")
    outputFile.write("colnames(bx2) = methodNameVec\n")
    outputFile.write("test_df2 <- w2l(bx2)\n")
    outputFile.write("bwplot(value ~ variable, data = test_df2, scales=list(tck=0, x=list(rot=45,cex=1.8), y=list(cex=1.5)), col = 'black',\n")
    outputFile.write("       main = list('"+cellTranslation[cell]+" / "+factorTranslation[factor]+"',cex=1.8), xlab = '', ylab = list('Absolute Distance of Predicted\\nLocations to TFBSs (bp)',cex=1.8),\n")
    outputFile.write("       par.settings = list( plot.symbol=list(col = 'black'),box.rectangle = list(col = 'black'),\n")
    outputFile.write("       box.dot = list(col = 'black'), box.umbrella= list(col = 'black')),\n")
    outputFile.write("       panel=function(...) {\n")
    outputFile.write("           panel.abline(h=c(0,20,40,60,80,100),lty=2,lwd=1.0,col='gray')\n")
    outputFile.write("           panel.bwplot(...)\n")
    outputFile.write("       })\n")
    outputFile.write("dev.off()\n")
    outputFile.write("toWrite = c('XXXXX',methodNameVec)\n")
    outputFile.write("for (i in (1:ncol(bx2))){\n")
    outputFile.write("  toWrite = c(toWrite,methodNameVec[i])\n")
    outputFile.write("  for (j in (1:ncol(bx2))){\n")
    outputFile.write("    if(i == j){\n")
    outputFile.write("      toWrite = c(toWrite,'XXXXX')\n")
    outputFile.write("    }\n")
    outputFile.write("    else{\n")
    outputFile.write("      teste = wilcox.test(bx2[,i],bx2[,j],paired=TRUE)\n")
    outputFile.write("      toWrite = c(toWrite,format(teste$p.value,digits=5))\n")
    outputFile.write("    }\n")
    outputFile.write("  }\n")
    outputFile.write("}\n")
    outputFile.write("write(toWrite,file='"+currOutHisLoc+factor+".txt',ncolumns=ncol(bx2)+1,append=FALSE,sep='\\t')\n")
    outputFile.write("system('epstopdf "+currOutHisLoc+factor+".eps')\n")
    outputFile.write("\n")

    # Writing tex script
    if(factorCounter == 4):
        outputTexFile.write("\\caption{Distribution of the distances from all true MPBSs to the center of the closest region predicted by segmentation-based methods for the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
        outputTexFile.write("\\label{fig:boxplot."+cell+"."+ev+"."+str(labelCounter)+"}\n")
        outputTexFile.write("\\end{figure}\n\n")
        outputTexFile.write("\\begin{figure}[h]\n")
        outputTexFile.write("\\centering\n")
        factorCounter = 0
        labelCounter += 1
    outputTexFile.write("    \\includegraphics[width=0.49\\textwidth]{"+factor+"}\n")
    #outputTexFile.write("    \\hspace{0.2cm}\n")
    factorCounter += 1

# Writing last tex lines
outputTexFile.write("\\caption{Distribution of the distances from all true MPBSs to the center of the closest region predicted by segmentation-based methods for the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
outputTexFile.write("\\label{fig:boxplot."+cell+"."+ev+"."+str(labelCounter)+"}\n")
outputTexFile.write("\\end{figure}\n\n")
outputTexFile.write("\\end{document}")

# Termination
outputFile.close()
outputTexFile.close()
os.system("R CMD BATCH "+outputFileName+" "+outputFileName+"OUT")
os.system("rm .RData")
os.system("cd "+currOutHisLoc+";pdflatex "+outputTexFileName)
os.system("rm "+outputFileName+"OUT "+outputTexFileName[:-3]+"aux "+outputTexFileName[:-3]+"log")



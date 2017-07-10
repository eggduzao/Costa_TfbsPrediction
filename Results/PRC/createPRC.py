#####################################################################################
# Script to create the R script that will create all the prc figures.
#####################################################################################

# Import
import os
import sys
import glob

# Input
cell = sys.argv[1] # Cell line label. Eg. H1hesc
prcFolder = sys.argv[2] # Folder to the prc curve results. Inside this folder the results must be separated by cell line.
outPrcLoc = sys.argv[3] # Output location of prc curves.

#################################################
# Parameters
#################################################

# Cell translation
cellFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/Cell.txt"
cellTranslation = dict()
cellFile = open(cellFileName,"r")
for line in cellFile:
    ll = line.strip().split("\t")
    cellTranslation[ll[0]] = ll[1]
cellFile.close()

# TF translation
factorFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/TF.txt"
factorTranslation = dict()
factorFile = open(factorFileName,"r")
for line in factorFile:
    ll = line.strip().split("\t")
    factorTranslation[ll[0]] = ll[1]
factorFile.close()

# Method translation
methodFileName = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/ROC_Method_"+cell+"_thesis.txt"
labelList = []; prcLabelList = []; fencyLabelList = []
labelFile = open(methodFileName,"r")
for line in labelFile:
    ll = line.strip().split("\t")
    labelList.append(ll[0])
    prcLabelList.append(ll[0]+"_REC"); prcLabelList.append(ll[0]+"_PRE");
    fencyLabelList.append(ll[1])
labelFile.close()

# prc-specific translations
legendYPosDict = dict([("H1hesc","1.06"),("HeLaS3","1.06"),("HepG2","1.06"),("Huvec","1.06"),("K562","1.06"),
   ("Mcf7","1.06"),("Saos2","1.06"),("LnCaP","1.06"),("m3134_with_DEX","1.06"),("m3134_wo_DEX","1.06")])
textYPosDict = dict([("H1hesc","1.03"),("HeLaS3","1.03"),("HepG2","1.03"),("Huvec","1.03"),("K562","1.03"),
   ("Mcf7","1.03"),("Saos2","1.03"),("LnCaP","1.03"),("m3134_with_DEX","1.03"),("m3134_wo_DEX","1.03")])
textYPosDictA = dict([("H1hesc","1.03"),("HeLaS3","1.03"),("HepG2","1.03"),("Huvec","1.03"),("K562","1.03"),
   ("Mcf7","1.03"),("Saos2","1.03"),("LnCaP","1.03"),("m3134_with_DEX","1.03"),("m3134_wo_DEX","1.03")])
textXPosDict = dict([("H1hesc","0.988"),("HeLaS3","0.988"),("HepG2","0.988"),("Huvec","0.988"),("K562","0.988"),
   ("Mcf7","0.988"),("Saos2","0.988"),("LnCaP","0.988"),("m3134_with_DEX","0.988"),("m3134_wo_DEX","0.988")])
styVecDict = dict([("H1hesc",["1","1","1","1","1","1","1","1","1","1","1","1","1","1"]),
                   ("HeLaS3",["1","1","1","1","1","1"]),
                   ("HepG2",["1","1","1","1","1","1"]),
                   ("Huvec",["1","1","1","1","1","1"]),
                   ("K562",["1","1","1","1","1","1","1","1","1","1","1","1","1","1"]),
                   ("Mcf7",["1","1","1","1","1","1"]),
                   ("Saos2",["1","1","1","1","1","1"]),
                   ("LnCaP",["1","1","1","1","1","1"]),
                   ("m3134_with_DEX",["1","1","1","1","1","1"]),
                   ("m3134_wo_DEX",["1","1","1","1","1","1"])])
rainbowDict = dict([("H1hesc","cols=rainbow("+str(len(styVecDict["H1hesc"]))+")"),
                   ("HeLaS3","cols=rainbow("+str(len(styVecDict["HeLaS3"]))+")"),
                   ("HepG2","cols=rainbow("+str(len(styVecDict["HepG2"]))+")"),
                   ("Huvec","cols=rainbow("+str(len(styVecDict["Huvec"]))+")"),
                   ("K562","cols=rainbow("+str(len(styVecDict["K562"]))+")"),
                   ("Mcf7","cols=rainbow("+str(len(styVecDict["HeLaS3"]))+")"),
                   ("Saos2","cols=rainbow("+str(len(styVecDict["HepG2"]))+")"),
                   ("LnCaP","cols=rainbow("+str(len(styVecDict["Huvec"]))+")"),
                   ("m3134_with_DEX","cols=rainbow("+str(len(styVecDict["HeLaS3"]))+")"),
                   ("m3134_wo_DEX","cols=rainbow("+str(len(styVecDict["HepG2"]))+")")])
myLegendDict = dict([
    ("H1hesc","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend7.R"),
    ("HeLaS3","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("HepG2","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("Huvec","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("K562","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend10.R"),
    ("Mcf7","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("Saos2","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("LnCaP","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("m3134_with_DEX","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R"),
    ("m3134_wo_DEX","/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/LabelTranslation/myLegend8.R")])


#################################################
# prc / Tex Initialization
#################################################

# Creating prc output
currOutPrcLoc = outPrcLoc+cell+"/"
os.system("mkdir -p "+currOutPrcLoc)
outputFileName = currOutPrcLoc+"script.R"
outputFile = open(outputFileName,"w")
outputTexFileName = currOutPrcLoc+"script.tex"
outputTexFile = open(outputTexFileName,"w")

# Writing global parameters to R script
outputFile.write("library(plotrix)\n")
outputFile.write(rainbowDict[cell]+"\n")
outputFile.write("styVec=c("+",".join(styVecDict[cell])+")\n")
outputFile.write("segLen=c(rep(2,"+str(len(styVecDict[cell]))+"),rep(.3,"+str(len(styVecDict[cell]))+"))\n")
outputFile.write("legendStyVec=c(styVec,rep(1,"+str(len(styVecDict[cell]))+"))\n")
outputFile.write("legendCol=c(cols,rep('white',"+str(len(styVecDict[cell]))+"))\n")
outputFile.write("mar.default <- c(5,5,4,2)\n")
outputFile.write("source('"+myLegendDict[cell]+"')\n")
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
finishJustReachedFlag = False

#################################################
# PRC / Tex Print
#################################################

# Iterating on factors
for prcFileName in glob.glob(prcFolder+cell+"/*_prc.txt"):

    finishJustReachedFlag = False

    # Fetching columns
    factor = "_".join(prcFileName.split("/")[-1].split("_")[:-1])

    if(factor == "POU5F1" or factor == "REST"): continue # Because FLR is not available

    prcFile = open(prcFileName,"r")
    headerList = prcFile.readline().strip().split("\t")
    posList = []
    newLabelList = []
    newFencyLabelList = []
    newPrcLabelList = []
    for i in range(0,len(prcLabelList)):
        if(prcLabelList[i] in headerList):
            posList.append(str(headerList.index(prcLabelList[i])+1))
            newPrcLabelList.append(prcLabelList[i])
            if(i%2==0):
                newLabelList.append(labelList[i/2])
                newFencyLabelList.append(fencyLabelList[i/2])
    prcFile.close()

    # Fetching AUC, sens & spec
    aucFileName = prcFolder+cell+"/"+factor+"_statsPR.txt"
    aucFile = open(aucFileName,"r")
    aucDict = dict()
    for line in aucFile:
        ll = line.strip().split("\t")
        if(ll[0] in newLabelList):
          aucDict[ll[0]] = ll[1] # AUPR ALL
    aucFile.close()
    aucList = []
    for k in newLabelList: aucList.append(aucDict[k])

    # Writing R script
    outputFile.write("# "+cell+" - "+factor+" \n")
    outputFile.write("prc=read.table('"+prcFileName+"',header=TRUE)\n")
    outputFile.write("postscript('"+currOutPrcLoc+factor+".eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')\n") # Save to eps
    outputFile.write("par(mar = mar.default)\n")
    outputFile.write("prc=prc[,c("+",".join(posList)+")]\n")
    outputFile.write("plot(prc[,1],prc[,2],type='n',ylab='Precision',xlab='Recall',main='"+cellTranslation[cell]+" / "+factorTranslation[factor]+"',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0,xlim=c(-0.01,1.01),ylim=c(-0.01,1.01))\n")
    outputFile.write("myLegend(0.7,"+legendYPosDict[cell]+",c("+",".join(["'"+e+"'" for e in newFencyLabelList])+","+",".join(["'"+str(round(float(e)*100,2))+"'" for e in aucList])+"),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=0.8,ncol=2,seg.len=segLen)\n")
    outputFile.write("text(x=0.853,y="+textYPosDict[cell]+",'METHOD',font=2,cex=0.8)\n")
    outputFile.write("text(x="+textXPosDict[cell]+",y="+textYPosDictA[cell]+",'AUPR',font=2,cex=0.8)\n")
    outputFile.write("for (i in (1:"+str(len(newLabelList))+")*2){\n")
    outputFile.write("  lines(prc[,i-1],prc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])\n")
    outputFile.write("}\n")
    outputFile.write("dev.off()\n")
    outputFile.write("system('epstopdf "+currOutPrcLoc+factor+".eps')\n")
    outputFile.write("\n")
    
    # Writing tex script
    if(factorCounter == 4):
        outputTexFile.write("\\caption{PRC curve created when applying the methods to data from the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
        outputTexFile.write("\\label{fig:prc."+cell+"."+str(labelCounter)+"}\n")
        outputTexFile.write("\\end{figure}\n\n")
        outputTexFile.write("\\begin{figure}[h]\n")
        outputTexFile.write("\\centering\n")
        factorCounter = 0
        labelCounter += 1
        finishJustReachedFlag = True
    outputTexFile.write("    \\includegraphics[width=0.49\\textwidth]{"+factor+"}\n")
    factorCounter += 1

# Writing last tex lines
if(not finishJustReachedFlag):
    outputTexFile.write("\\caption{PRC curve created when applying the methods to data from the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
    outputTexFile.write("\\label{fig:prc."+cell+"."+str(labelCounter)+"}\n")
    outputTexFile.write("\\end{figure}\n\n")
outputTexFile.write("\\end{document}")

# Termination
outputFile.close()
outputTexFile.close()
os.system("R CMD BATCH "+outputFileName+" "+outputFileName+"OUT")
os.system("rm .RData")
#os.system("cd "+currOutRocLoc+";pdflatex "+outputTexFileName)
#os.system("rm "+outputFileName+"OUT "+outputTexFileName[:-3]+"aux "+outputTexFileName[:-3]+"log "+outputTexFileName)
os.system("rm "+outputFileName+"OUT "+outputTexFileName)



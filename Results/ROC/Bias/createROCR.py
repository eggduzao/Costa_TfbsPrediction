#####################################################################################
# Script to create the R script that will create all the ROC figures.
#####################################################################################

# Import
import os
import sys
import glob

# Input
cell = sys.argv[1] # Cell line label. Eg. H1hesc
rocFolder = sys.argv[2] # Folder to the ROC curve results. Inside this folder the results must be separated by cell line.
outRocLoc = sys.argv[3] # Output location of ROC curves.

#################################################
# Parameters
#################################################

# Cell translation
cellFileName = "/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/Cell.txt"
cellTranslation = dict()
cellFile = open(cellFileName,"r")
for line in cellFile:
    ll = line.strip().split("\t")
    cellTranslation[ll[0]] = ll[1]
cellFile.close()

# TF translation
factorFileName = "/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/TF.txt"
factorTranslation = dict()
factorFile = open(factorFileName,"r")
for line in factorFile:
    ll = line.strip().split("\t")
    factorTranslation[ll[0]] = ll[1]
factorFile.close()

# Method translation
methodFileName = "/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/ROC_Method_"+cell+".txt"
labelList = []; rocLabelList = []; fencyLabelList = []; texLabelList = []
labelFile = open(methodFileName,"r")
for line in labelFile:
    ll = line.strip().split("\t")
    labelList.append(ll[0])
    rocLabelList.append(ll[0]+"_FPR"); rocLabelList.append(ll[0]+"_TPR");
    fencyLabelList.append(ll[1])
    texLabelList.append(ll[2])
labelFile.close()

# ROC-specific translations
legendYPosDict = dict([("H1hesc","0.54"),("HeLaS3","0.325"),("HepG2","0.325"),("Huvec","0.325"),("K562","0.54")])
textYPosDict = dict([("H1hesc","0.50"),("HeLaS3","0.282"),("HepG2","0.282"),("Huvec","0.282"),("K562","0.50")])
textYPosDictA = dict([("H1hesc","0.4955"),("HeLaS3","0.278"),("HepG2","0.278"),("Huvec","0.278"),("K562","0.4955")])
textXPosDict = dict([("H1hesc","0.41"),("HeLaS3","0.405"),("HepG2","0.405"),("Huvec","0.405"),("K562","0.41")])
styVecDict = dict([("H1hesc",["1","1","1","1","1","1","1","1","1"]),
                   ("HeLaS3",["1","1","1","1","1"]),
                   ("HepG2",["1","1","1","1","1"]),
                   ("Huvec",["1","1","1","1","1"]),
                   ("K562",["1","1","1","1","1","1","1","1","1"])])
"""
pointStyVecDict = dict([("H1hesc",["21","21","21","21","21","21","21","21","21"]),
                   ("HeLaS3",["21","21","21","21","21"]),
                   ("HepG2",["21","21","21","21","21"]),
                   ("Huvec",["21","21","21","21","21"]),
                   ("K562",["21","21","21","21","21","21","21","21","21"])])
"""
rainbowDict = dict([("H1hesc","cols=rainbow("+str(len(styVecDict["H1hesc"]))+")"),
                   ("HeLaS3","cols=rainbow("+str(len(styVecDict["HeLaS3"]))+")"),
                   ("HepG2","cols=rainbow("+str(len(styVecDict["HepG2"]))+")"),
                   ("Huvec","cols=rainbow("+str(len(styVecDict["Huvec"]))+")"),
                   ("K562","cols=rainbow("+str(len(styVecDict["K562"]))+")")])
myLegendDict = dict([
    ("H1hesc","/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend3.R"),
    ("HeLaS3","/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R"),
    ("HepG2","/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R"),
    ("Huvec","/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend4.R"),
    ("K562","/home/egg/Projects/TfbsPrediction/Results/LabelTranslation/myLegend3.R")])


#################################################
# ROC / Tex Initialization
#################################################

# Creating ROC output
currOutRocLoc = outRocLoc+cell+"/"
os.system("mkdir -p "+currOutRocLoc)
outputFileName = currOutRocLoc+"script.R"
outputFile = open(outputFileName,"w")
outputTexFileName = currOutRocLoc+"script.tex"
outputTexFile = open(outputTexFileName,"w")

# Writing global parameters to R script
outputFile.write("library(plotrix)\n")
outputFile.write(rainbowDict[cell]+"\n")
outputFile.write("styVec=c("+",".join(styVecDict[cell])+")\n")
#outputFile.write("pointStyVec=c("+",".join(pointStyVecDict[cell])+")\n")
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
# ROC / Tex Print
#################################################

# Iterating on factors
for rocFileName in glob.glob(rocFolder+cell+"/*_roc.txt"):

    finishJustReachedFlag = False

    # Fetching columns
    factor = "_".join(rocFileName.split("/")[-1].split("_")[:-1])
    rocFile = open(rocFileName,"r")
    headerList = rocFile.readline().strip().split("\t")
    posList = []
    newLabelList = []
    newFencyLabelList = []
    newRocLabelList = []
    newTexLabelList = []
    for i in range(0,len(rocLabelList)):
        if(rocLabelList[i] in headerList):
            posList.append(str(headerList.index(rocLabelList[i])+1))
            newRocLabelList.append(rocLabelList[i])
            if(i%2==0):
                newLabelList.append(labelList[i/2])
                newFencyLabelList.append(fencyLabelList[i/2])
                newTexLabelList.append(texLabelList[i/2])
    rocFile.close()

    # Fetching AUC, sens & spec
    aucFileName = rocFolder+cell+"/"+factor+"_stats.txt"
    aucFile = open(aucFileName,"r")
    aucDict = dict(); texDict = dict()
    for line in aucFile:
        ll = line.strip().split("\t")
        if(ll[0] in newLabelList):
          aucDict[ll[0]] = ll[2] # 10% FPR AUC
          texDict[ll[0]] = [ll[3],ll[4]] # 10% FPR Sens & Spec
    aucFile.close()
    aucList = []
    for k in newLabelList: aucList.append(aucDict[k])
    texList = []
    for k in newLabelList:
        if(k=="PWM"): texList.append("NA"); texList.append("NA")
        else: texList.append(texDict[k][1]); texList.append(texDict[k][0])

    # Writing R script
    outputFile.write("# "+cell+" - "+factor+" \n")
    outputFile.write("roc=read.table('"+rocFileName+"',header=TRUE)\n")
    outputFile.write("postscript('"+currOutRocLoc+factor+".eps',width=7.0,height=7.0,horizontal=FALSE,paper='special')\n") # Save to eps
    outputFile.write("par(mar = mar.default)\n")
    outputFile.write("roc=roc[,c("+",".join(posList)+")]\n")
    outputFile.write("plot(roc[,1],roc[,2],type='n',ylab='Sensitivity',xlab='Specificity',main='"+cellTranslation[cell]+" / "+factorTranslation[factor]+"',cex.lab=2.0,cex.axis=1.5,cex.main=1.7,cex.sub=2.0)\n")
    outputFile.write("myLegend(-0.035,"+legendYPosDict[cell]+",c("+",".join(["'"+e+"'" for e in newFencyLabelList])+","+",".join(["'"+str(round(float(e)*100,2))+"'" for e in aucList])+"),lty=legendStyVec,col=legendCol,lwd=2.0,title='',cex=1.3,ncol=2,seg.len=segLen)\n")
    outputFile.write("text(x=0.208,y="+textYPosDict[cell]+",'METHOD',font=2,cex=1.3)\n")
    outputFile.write("text(x="+textXPosDict[cell]+",y="+textYPosDictA[cell]+",'AUC(%)',font=2,cex=1.3)\n")
    #outputFile.write("pointList=c("+",".join(texList)+")\n")
    #outputFile.write("lines(c(0.0,1.0),c(1.0,0.0),col='#CCCCCC',lwd=0.5,lty=2)\n")
    outputFile.write("for (i in (1:"+str(len(newLabelList))+")*2){\n")
    outputFile.write("  lines(1-roc[,i-1],roc[,i],col=cols[i/2],lwd=2.0,lty=styVec[i/2])\n")
    #outputFile.write("  points(pointList[i-1],pointList[i],col=cols[i/2],lwd=3.0,pch=pointStyVec[i/2],cex=2.0)\n")
    outputFile.write("}\n")
    #outputFile.write("legend(-0.035,"+legendYPosDict[cell]+",c("+",".join(["'"+newFencyLabelList[i]+" = "+str(round(float(aucList[i]),4))+"'" for i in range(0,len(newFencyLabelList))])+"),lty=styVec,col=cols,lwd=1.0,title='AUC')\n")
    outputFile.write("dev.off()\n")
    outputFile.write("system('epstopdf "+currOutRocLoc+factor+".eps')\n")
    outputFile.write("\n")
    
    # Writing tex script
    if(factorCounter == 4):
        outputTexFile.write("\\caption{ROC curve created when applying the methods to data from the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
        outputTexFile.write("\\label{fig:roc."+cell+"."+str(labelCounter)+"}\n")
        outputTexFile.write("\\end{figure}\n\n")
        outputTexFile.write("\\begin{figure}[h]\n")
        outputTexFile.write("\\centering\n")
        factorCounter = 0
        labelCounter += 1
        finishJustReachedFlag = True
    outputTexFile.write("    \\includegraphics[width=0.49\\textwidth]{"+factor+"}\n")
    #outputTexFile.write("    \\hspace{0.2cm}\n")
    factorCounter += 1

# Writing last tex lines
if(not finishJustReachedFlag):
    outputTexFile.write("\\caption{ROC curve created when applying the methods to data from the cell line "+cellTranslation[cell]+". All HMM Models were trained with data from H1-hESC and K562 cell lines.}\n")
    outputTexFile.write("\\label{fig:roc."+cell+"."+str(labelCounter)+"}\n")
    outputTexFile.write("\\end{figure}\n\n")
outputTexFile.write("\\end{document}")

# Termination
outputFile.close()
outputTexFile.close()
os.system("R CMD BATCH "+outputFileName+" "+outputFileName+"OUT")
os.system("rm .RData")
os.system("cd "+currOutRocLoc+";pdflatex "+outputTexFileName)
os.system("rm "+outputFileName+"OUT "+outputTexFileName[:-3]+"aux "+outputTexFileName[:-3]+"log")



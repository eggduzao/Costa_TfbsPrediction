
# Import
import os
import sys

# Input WIG
dnaseFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/K562/DNase.bw"
dnaseBOFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/K562/DNaseBO.bw"
dnaseNormFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/K562/InputSignal/hd/DNase_norm.bw"
dnaseSlopeFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/K562/InputSignal/hd/DNase_slope.bw"
histoneFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Counts/K562/H3K4me3.bw"
histoneNormFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/K562/InputSignal/hd/H3K4me3_norm.bw"
histoneSlopeFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/HMM/K562/InputSignal/hd/H3K4me3_slope.bw"

# Input BED
predictionFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/Footprints/Predictions/K562/DH_HMM_HD/H3K4me3_ModelK562_tab.bed"
regionFileName = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/SignalStatistics/AssymetricSample/region.bed"

# Input
outputLocation = "/hpcwork/izkf/projects/egg/TfbsPrediction/Results/SignalStatistics/AssymetricSample/Input/"
def outputFileName(myFileName):
    return outputLocation+myFileName.split("/")[-1]

# Cropping bed file
os.system("intersectBed -wa -a "+predictionFileName+" -b "+regionFileName+" > "+outputFileName(predictionFileName))

# Cropping wig files
wigFileList = [dnaseFileName,dnaseBOFileName,dnaseNormFileName,dnaseSlopeFileName,histoneFileName,histoneNormFileName,histoneSlopeFileName]
for wigFileName in wigFileList: os.system("cropWig "+wigFileName+" "+regionFileName+" "+outputLocation)



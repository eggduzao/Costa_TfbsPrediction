
# Import
import os

###################################################################################################
# DC
###################################################################################################

# Parameters
organism="--organism mm9"
fpr="--fpr 0.0001"
precision="--precision 10000"
pseudocounts="--pseudocounts 0.1"
rand_proportion="--rand-proportion 0.001"
output_location="--output-location /hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/motifs/mm9/"
bigbed="--bigbed"
inputMatrix="/hpcwork/izkf/projects/TfbsPrediction/Data/NatMetReply/motifs/exp_mat.txt"

# Execution
myL = "MM"
clusterCommand = "bsub "
clusterCommand += "-J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
clusterCommand += "-W 10:00 -M 8192 -S 100 -R \"select[hpcwork]\" ./motifAnalysis_pipeline.zsh --matching "
clusterCommand += organism+" "+fpr+" "+precision+" "+pseudocounts+" "+rand_proportion+" "
clusterCommand += output_location+" "+bigbed+" "+inputMatrix
os.system(clusterCommand)
# -P izkf



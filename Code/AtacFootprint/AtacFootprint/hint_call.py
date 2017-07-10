
# Import
import os

# Direction Loop
extBothList = [True]
for eb in extBothList:

  # Extension Loop
  extList = ["1"]
  for ext in extList:

    # Parameters
    if(eb):
      ext_both_directions = "--ext-both-directions"
      oname = "twoside"
    else:
      ext_both_directions = ""
      oname = "oneside"
    dnase_norm_per = "--dnase-norm-per=99"
    dnase_slope_per = "--dnase-slope-per=99"
    dnase_frag_ext = "--dnase-frag-ext="+ext
    hmm_file = "--hmm-file=/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Results/ATAC/HMM/Models_ATAC/K562/Model/regular.hmm"
    inputMatrix = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/AtacFootprint/exp_mat/scK562.txt"
    organism = "--organism=hg19"
    outLoc = "/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/AtacFootprint/results_scK562/"+oname+"_"+ext+"/"
    os.system("mkdir -p "+outLoc)
    output_location = "--output-location="+outLoc

    # Execution
    #myL = "_".join(["ATAC"])
    #clusterCommand = "bsub -J "+myL+" -o "+myL+"_out.txt -e "+myL+"_err.txt "
    #clusterCommand += "-W 100:00 -M 12000 -S 100 -P izkf -R \"select[hpcwork]\" ./hint_pipeline.zsh "
    #clusterCommand += organism+" "+output_location+" "+ext_both_directions+" "+hmm_file+" "
    #clusterCommand += dnase_norm_per+" "+dnase_slope_per+" "+dnase_frag_ext+" "+inputMatrix
    #os.system(clusterCommand)

    clusterCommand = "rgt-hint "+organism+" "+output_location+" "+ext_both_directions+" "+hmm_file+" "
    clusterCommand += dnase_norm_per+" "+dnase_slope_per+" "+dnase_frag_ext+" "+inputMatrix
    os.system(clusterCommand)



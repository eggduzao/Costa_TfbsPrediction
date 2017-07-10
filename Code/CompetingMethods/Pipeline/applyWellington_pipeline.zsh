#!/bin/zsh

# Export
module load DEVELOP python/2.7.5 &> /dev/null
export PATH=$PATH:/home/eg474423/Installation/meme/bin # MEME SUITE
export PATH=$PATH:/home/eg474423/Installation/HINT/bin # HINT
export PATH=$PATH:/home/eg474423/Installation/motifanalysis/bin # Motif Analysis
export PATH=$PATH:/home/eg474423/Installation/wiggle_to_psp_utilities # Cuellar Method
export PATH=$PATH:/hpcwork/izkf/bin
export PATH=$PATH:/hpcwork/izkf/opt/bin
export PATH=$PATH:/home/eg474423/perl/bin
export PATH=$PATH:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/bin
export PYTHONPATH=$PYTHONPATH:/home/eg474423/Installation/pysam-0.8.3/lib/python2.7/site-packages # Pysam 0.8.3
export PYTHONPATH=$PYTHONPATH:/home/eg474423/Installation/HINT/lib/python2.7/site-packages # HINT
export PYTHONPATH=$PYTHONPATH:/home/eg474423/Installation/motifanalysis/lib/python2.7/site-packages # Motif Analysis
export PYTHONPATH=$PYTHONPATH:/home/eg474423/Installation/bx-python-0.7.3/lib/python2.7/site-packages # bx-python-0.7.3
export PYTHONPATH=$PYTHONPATH:/home/eg474423/Installation/pyDNase-0.1.7/lib/python2.7/site-packages/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib/python2.7/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/opt/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/opt/lib64
export R_LIBS_USER=$R_LIBS_USER:/home/eg474423/R

applyWellington $1 $2 $3



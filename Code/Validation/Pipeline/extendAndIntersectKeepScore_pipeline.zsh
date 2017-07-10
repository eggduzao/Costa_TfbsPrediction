#!/bin/zsh

export PATH=/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/bin/:/home/eg474423/app/bin/:/home/eg474423/meme/bin/:$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/
export PYTHONPATH=/home/eg474423/app/lib/python2.6/:/home/eg474423/app/lib/python2.6/site-packages/:/home/eg474423/app/lib64/python2.6/:/home/eg474423/app/lib64/python2.6/site-packages/:$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=/home/eg474423/app/lib/:/home/eg474423/app/lib64/:$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

leftExt=$1
rightExt=$2
mpbsFileName=$3
predFileName=$4
outputLocation=$5

extendAndIntersectKeepScore $leftExt $rightExt $mpbsFileName $predFileName $outputLocation



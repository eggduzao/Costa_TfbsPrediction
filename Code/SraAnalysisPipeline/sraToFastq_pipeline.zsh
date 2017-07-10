#!/bin/zsh

# Export
export PYTHONPATH=$PYTHONPATH
export PATH=$PATH:/hpcwork/izkf/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

cd $2
fastq-dump.2.5.0 $1



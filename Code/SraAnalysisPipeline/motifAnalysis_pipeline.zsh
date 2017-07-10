#!/bin/zsh

# This script is running the RGT version installed in my home directory
module load DEVELOP python/2.7.5

export PYTHONPATH=/home/eg474423/app_motifanalysis/lib/python2.7/site-packages/:/usr/local_rwth/sw/python/2.7.5/x86_64/extra/lib/python2.7/site-packages:/hpcwork/izkf/lib/python2.7/site-packages/

export PATH=/home/eg474423/app_motifanalysis/bin/:/usr/local_rwth/sw/python/2.7.5/x86_64/bin:/usr/local_rwth/sw/python/2.7.5/x86_64/extra/bin:/opt/MPI/bin:/opt/MPI/openmpi-1.6.5/linux/intel/bin:/opt/intel/Compiler/14.0/3.174/rwthlnk/bin/intel64:/opt/intel/Compiler/14.0/3.174/rwthlnk/bin/sourcechecker/bin/intel64:/opt/lsf/9.1/linux2.6-glibc2.3-x86_64/etc:/opt/lsf/9.1/linux2.6-glibc2.3-x86_64/bin:/usr/local_host/sbin:/usr/local_host/bin:/usr/local_rwth/sbin:/usr/local_rwth/bin:/usr/bin:/usr/sbin:/sbin:/usr/dt/bin:/usr/bin/X11:/usr/java/bin:/usr/kerberos/sbin:/usr/kerberos/bin:/usr/local/bin:/usr/local/sbin:/opt/csw/bin:.:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/Code/bin/:/home/eg474423/app/bin/:/home/eg474423/meme/bin/

export LD_LIBRARY_PATH=/usr/local_rwth/sw/python/2.7.5/x86_64/lib:/usr/local_rwth/sw/python/2.7.5/x86_64/extra/lib:/usr/local_rwth/sw/python/2.7.5/x86_64/extra/CBLAS:/opt/MPI/openmpi-1.6.5/linux/intel/lib:/opt/MPI/openmpi-1.6.5/linux/intel/lib/lib32:/opt/intel/mic/myo/lib:/opt/intel/mic/coi/host-linux-release/lib:/opt/intel/Compiler/14.0/3.174/rwthlnk/ipp/lib/intel64:/opt/intel/Compiler/14.0/3.174/rwthlnk/ipp/lib/ia32:/opt/intel/Compiler/14.0/3.174/rwthlnk/mkl/lib/intel64:/opt/intel/Compiler/14.0/3.174/rwthlnk/mkl/lib/ia32:/opt/intel/Compiler/14.0/3.174/rwthlnk/mkl/lib/mic:/opt/intel/Compiler/14.0/3.174/rwthlnk/tbb/lib/intel64:/opt/intel/Compiler/14.0/3.174/rwthlnk/tbb/lib/ia32:/opt/intel/Compiler/14.0/3.174/rwthlnk/tbb/lib/mic:/opt/intel/Compiler/14.0/3.174/rwthlnk/compiler/lib/intel64:/opt/intel/Compiler/14.0/3.174/rwthlnk/compiler/lib/ia32:/opt/intel/Compiler/14.0/3.174/rwthlnk/compiler/lib/mic:/opt/lsf/9.1/linux2.6-glibc2.3-x86_64/lib:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/:/home/eg474423/app/lib/:/home/eg474423/app/lib64/

allParameters=($argv[1,$#argv])
rgt-motifanalysis $allParameters



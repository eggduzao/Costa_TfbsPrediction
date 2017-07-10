#!/bin/zsh

# Imports
export PATH=$PATH:/hpcwork/izkf/bin/:/hpcwork/izkf/opt/bin/:/work/eg474423/eg474423_Projects/trunk/TfbsPrediction/exp/bin/
export PYTHONPATH=$PYTHONPATH:/hpcwork/izkf/lib64/python2.6/:/hpcwork/izkf/lib64/python2.6/site-packages/:/hpcwork/izkf/lib/python2.6/:/hpcwork/izkf/lib/python2.6/site-packages/:/hpcwork/izkf/opt/lib64/python2.6/:/hpcwork/izkf/opt/lib64/python2.6/site-packages/:/hpcwork/izkf/opt/lib/python2.6/:/hpcwork/izkf/opt/lib/python2.6/site-packages/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hpcwork/izkf/lib64/:/hpcwork/izkf/lib/:/hpcwork/izkf/opt/lib/:/hpcwork/izkf/opt/lib64/:/hpcwork/izkf/opt/lib/

# Parameters
folderName=$1 # WITH THE FINAL SLASH (/)
chromLoc=$2 # Chrom sizes file for specific organism

# Parameter processing
lastName=(${(s:/:)folderName})
lastName=$lastName[-1]

# 1. Gunzip gz files
treatList=($folderName$lastName"_MACS_wiggle/treat/"*)
controlList=($folderName$lastName"_MACS_wiggle/control/"*)
if [[ -z $treatList ]]; then
    print "treatList is empty."
else
    gunzip $folderName$lastName"_MACS_wiggle/treat/"*
fi
if [[ -z $controlList ]]; then
    print "controlList is empty."
else
    gunzip $folderName$lastName"_MACS_wiggle/control/"*
fi

# 2. Remove genome browser track lines (first line) from files
treatList=($folderName$lastName"_MACS_wiggle/treat/"*)
controlList=($folderName$lastName"_MACS_wiggle/control/"*)
if [[ -z $treatList ]]; then
    print "treatList is empty."
else
    for fn in $treatList
    do
        sed '1d' $fn > $fn[1,-5]"_temp.wig"
        rm $fn
    done
fi
if [[ -z $controlList ]]; then
    print "controlList is empty."
else
    for fn in $controlList
    do
        sed '1d' $fn > $fn[1,-5]"_temp.wig"
        rm $fn
    done
fi

# 3. Merge files
treatList=($folderName$lastName"_MACS_wiggle/treat/"*)
controlList=($folderName$lastName"_MACS_wiggle/control/"*)
if [[ -z $treatList ]]; then
    print "treatList is empty."
else
    cat $treatList > $folderName$lastName"_treat_signal.wig"
fi
if [[ -z $controlList ]]; then
    print "controlList is empty."
else
    cat $controlList > $folderName$lastName"_control_signal.wig"
fi
rm -rf $folderName$lastName"_MACS_wiggle/"

# 4. Convert to bw
for fn in $folderName*"_signal.wig"
do
    wigToBigWig $fn $chromLoc $fn[1,-4]"bw"
done
rm $folderName*"_signal.wig"



#!/bin/zsh

mpbsFileName="DU_H1hesc.bed"
fpFileName="DU_H1hesc_HINTBC.bed"
outputLocation="./"

python chipValidation.py $mpbsFileName $fpFileName $outputLocation



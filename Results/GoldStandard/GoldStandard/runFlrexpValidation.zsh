#!/bin/zsh

expFileName="TF_Expression_H1hesc_K562.txt"
mpbsFileName1="DU_H1hesc_FLR.bed"
mpbsFileName2="DU_K562_FLR.bed"
fpFileName1="DU_H1hesc_HINTBC.bed"
fpFileName2="DU_K562_HINTBC.bed"
outputLocation="./"

python flrexpValidation.py $expFileName $mpbsFileName1 $mpbsFileName2 $fpFileName1 $fpFileName2 $outputLocation

python flrexpValidation.py TF_Expression_H1hesc_K562.txt DU_H1hesc_FLR.bed DU_K562_FLR.bed DU_H1hesc_HINTBC.bed DU_K562_HINTBC.bed ./

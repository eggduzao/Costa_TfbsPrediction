#!/bin/zsh
predictionHistogram y 30 CTCF mpbs.bed pred1,pred2 pred1.bed,pred2.bed ./
predictionHistogram y 30 GABP mpbs.bed pred1,pred2 pred1.bed,pred2.bed ./
predictionHistogram y 30 all mpbs.bed pred1,pred2 pred1.bed,pred2.bed ./

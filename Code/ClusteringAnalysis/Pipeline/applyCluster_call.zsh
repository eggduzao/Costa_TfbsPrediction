#!/bin/zsh

# Global Parameters
set noClust = "2"
set inputLocation = "/home/egg/Project_TFBS/Results/Cluster/K562/Post/Raw/"
set outputLocationStat = "/home/egg/Project_TFBS/Results/Cluster/K562/Stat/"
set outputLocationPost = "/home/egg/Project_TFBS/Results/Cluster/K562/Post/Clust/"
set outputLocationHeat = "/home/egg/Project_TFBS/Results/Cluster/K562/Heat/Clust/"

# Variations
set methodK = ( "mean" "median" )
set methodH = ( "sl" "cl" "el" "al" )
set distance = ( "corr" "abscorr" "uncentcorr" "absunccorr" "spearman" "kendall" "euc" "cityblock" )

set tempCwd = $cwd
cd "/home/egg/Project_TFBS/Results/MPBS/All/"
set factorList = ( * )
cd $tempCwd

# Execution
foreach fn ( $factorList )
    foreach d ( $distance )
        foreach m ( $methodK )
            ./applyCluster_pipeline.csh "k" $m $d $noClust $inputLocation$fn".post" $outputLocationStat $outputLocationPost $outputLocationHeat &
        end
        foreach m ( $methodH )
            ./applyCluster_pipeline.csh "h" $m $d $noClust $inputLocation$fn".post" $outputLocationStat $outputLocationPost $outputLocationHeat &
        end
    end
    sleep 2000
end



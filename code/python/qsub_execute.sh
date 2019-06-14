#!/bin/bash
source miniconda3/bin/activate pytorch
cd ~
python $SCRIPT --name $NAME --variant $VARIANT --method $METHOD --cv $CV

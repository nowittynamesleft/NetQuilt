#!/bin/sh
# Input taxonomy IDs as comma-delimited argument, e.g. 9606,10010,511145
python step_1.py $1 0.005 0.05;
python step_2.py $1;
python step_3.py $1 0.6;
echo "Species input"
echo "$1"
echo "Finished steps 1-3, preprocessing finished. Run multispecies.py in NetQuilt/scripts/model_scripts directory."

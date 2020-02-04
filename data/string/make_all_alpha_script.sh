#!/bin/bash
# Generates preprocessing script for alphas 0.0 - 1.0 in 0.1 increments for taxa delimited by , as the first argument

if [ -f $1_generate_all_alpha_preprocessing.sh ]; then 
    echo $1_generate_all_alpha_preprocessing.sh
    rm $1_generate_all_alpha_preprocessing.sh
    rm "$1"_*_logfile.txt
fi

touch "$1_generate_all_alpha_preprocessing.sh"

echo "python pipeline.py $1 0.0 > $1_0.0logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.1 > $1_0.1logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.2 > $1_0.2logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.3 > $1_0.3logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.4 > $1_0.4logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.5 > $1_0.5logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.6 > $1_0.6logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.7 > $1_0.7logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.8 > $1_0.8logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 0.9 > $1_0.9logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"
echo "python pipeline.py $1 1.0 > $1_1.0logfile.txt" >> "$1_generate_all_alpha_preprocessing.sh"

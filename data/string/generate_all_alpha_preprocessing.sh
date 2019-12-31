#!/bin/bash
# Runs preprocessing pipeline for alphas 0.0 - 1.0 in 0.1 increments for taxa delimited by , as the first argument

python pipeline.py $1 0.0
python pipeline.py $1 0.1
python pipeline.py $1 0.2
python pipeline.py $1 0.3
python pipeline.py $1 0.4
python pipeline.py $1 0.5
python pipeline.py $1 0.6
python pipeline.py $1 0.7
python pipeline.py $1 0.8
python pipeline.py $1 0.9
python pipeline.py $1 1.0

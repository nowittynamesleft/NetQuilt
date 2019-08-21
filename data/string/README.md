This folder contains scripts that handle fetching data from string and running blastp in order to preprocess the data for dimensionality reduction and classification steps that are handled in the scripts/model_scripts folder.

In order to create the random walk with restarts profiles the off diagonal block matrices (the interspecies connections based on blast e-values) to create the input S matrix to the autoencoder, run:

python pipeline.py {taxonomy ids delimited by commas} {alpha}

For example, to get the RWR profiles and diagonal block matrices for the bacteria with taxonomy ids 511145, 316407, 316385, 224308, 71421, and 243273, with an alpha value of 0.6, as well as a large annotation matrix containing GO terms for these organisms given the "all_go_knowledge_full.tsv" file from version 10.5 of the STRING database, run:

python pipeline.py 511145,316407,316385,224308,71421,243273 0.6

The "generate_all_alpha_preprocessing.sh" shell script just runs the pipeline script for a range of alphas (0.0 to 1.0 with increments of 0.1) for the organisms you input as the first argument, also delimited by commas, as in the following command to generate data for different alphas for the bacteria above:

./generate_all_alpha_preprocessing.sh 511145,316407,316385,224308,71421,243273

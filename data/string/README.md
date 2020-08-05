This folder contains scripts that handle fetching data from string and running blastp in order to preprocess the data for classification that is handled in the scripts/model_scripts folder.

In order to create the off diagonal IsoRank matrices (with the interspecies connections based on blast e-values) to create the input S matrix to the autoencoder, run:

python pipeline.py {taxonomy ids delimited by commas} {alpha}

For example, to get the RWR profiles and diagonal block matrices for the bacteria with taxonomy ids 511145, 316407, 316385, 224308, 71421, and 243273, with an alpha value of 0.6, as well as a large annotation matrix containing GO terms for these organisms given the "all_go_knowledge_full.tsv" file from version 10.5 of the STRING database, run:

python pipeline.py 511145,316407,316385,224308,71421,243273 0.6

The "generate_all_alpha_preprocessing.sh" shell script just runs the pipeline script for a range of alphas (0.0 to 1.0 with increments of 0.1) for the organisms you input as the first argument, also delimited by commas, as in the following command to generate data for different alphas for the bacteria above:

./generate_all_alpha_preprocessing.sh 511145,316407,316385,224308,71421,243273

To run the resulting "511145,316407,316385,224308,71421,243273_generate_all_alpha_preprocessing.sh" script, use disBatch (make sure to first "module add disBatch" on the Flatiron cluster). This is not required, you can use any distributed computation program (see https://github.com/flatironinstitute/disBatch to have an idea of what features such a program might have) to run the pipeline.

sbatch -N 4 -p ccb --ntasks-per-node 3 --exclusive --wrap "disBatch.py 511145,316407,316385,224308,71421,243273_generate_all_alpha_preprocessing.sh"

---------- Sample Use Case ----------

Let's say you're interested in using annotations and networks from some well-studied bacteria to help annotate Prevotella copri (tax id 537011).
Here is what you need to run:

With a normal multi-CPU node:
Step 1:
Download networks, fastas, annotations from STRING:
    python step_1.py 511145,316407,316385,224308,71421,243273,537011
Step 2:
Compute BLAST between species:
    python step_2.py 511145,316407,316385,224308,71421,243273,537011
Step 3:
Compute IsoRank for all pairs of species using the BLAST files and each species' network file for a given alpha value (0.6 works well for bacterial networks in our study):
    python step_3.py 511145,316407,316385,224308,71421,243273,537011 0.6

With a GPU node:
Step 4:
Run multispecies maxout neural network on the IsoRank matrices with annotations as labels, and output predictions for all proteins.
    (in NetQuilt/scripts/model_scripts/ directory)
    python multispecies.py --tax_ids 511145,316407,316385,224308,71421,243273,537011 --valid_type full_prediction --model_name bacteria_including_prevotella 
    --results_path ../results/test/ --data_folder ../../data/string/ --alpha 0.6 --annot ../../data/string/string_annot/511145-316407-316385-224308-71421-243273-537011_string.01_2019_annotations.pckl 
    --ont molecular_function --test_goid_fname ../../data/string/string_annot/511145-316407-316385-224308-71421-243273-537011_molecular_function_train_goids.pckl 
    --use_orig_features --use_nn_val --isorank_diag

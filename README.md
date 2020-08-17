This is the repository for NetQuilt, a multispecies network-based protein function prediction method incorporating homology information with IsoRank.

In the ./data/string/ directory, there are scripts used to download and preprocess data from STRING used in the method.

In the ./scripts/model_scripts directory, there are scripts used to run the method once IsoRank has been run on all pairs of networks.

multispecies.py is the main file for running the model after the preprocessing steps. Run python multispecies.py -h for options and use instructions.

Uses tensorflow version 1.14.

---------- Sample Use Case ----------

Let's say you're interested in using annotations and networks from some well-studied bacteria to help annotate Prevotella copri (tax id 537011).

Here is what you need to run:

With a normal multi-CPU node, in NetQuilt/data/string directory:

Step 1:

Download networks, fastas, annotations from STRING. Selected GO terms annotate at minimum 0.5% (proportion 0.005) of all proteins in selected networks, and maximum 5% (proportion 0.05) all proteins in selected networks:

python step_1.py 511145,316407,224308,71421,243273,537011 0.005 0.05

Step 2:

Compute BLAST between species:

python step_2.py 511145,316407,224308,71421,243273,537011

Step 3:

Compute IsoRank for all pairs of species using the BLAST files and each species' network file for a given alpha value (0.6 works well for bacterial networks in our study):

python step_3.py 511145,316407,224308,71421,243273,537011 0.6

With a GPU node, in NetQuilt/scripts/model_scripts/ directory:

Step 4:

Run multispecies maxout neural network on the IsoRank matrices with annotations as labels, and output predictions for all proteins.

python multispecies.py --tax_ids 511145,316407,224308,71421,243273,537011 --valid_type full_prediction --model_name bacteria_including_prevotella 
    --results_path ../results/test/ --data_folder ../../data/string/ --alpha 0.6 --annot ../../data/string/string_annot/511145-316407-224308-71421-243273-537011_string.01_2019_annotations.pckl 
    --ont molecular_function --test_goid_fname ../../data/string/string_annot/511145-316407-224308-71421-243273-537011_molecular_function_train_goids.pckl 
    --use_orig_features --use_nn_val --isorank_diag --arch_set bac

The results file that this step produces will be found at this path: ./scripts/results/test/bacteria_including_prevotella_alpha_0.6_use_nn_molecular_function_pred_file_complete.pckl
This file contains the full prediction matrix for the GO terms chosen by the thresholds chosen in the first step.

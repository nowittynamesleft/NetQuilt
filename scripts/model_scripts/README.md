For running multispecies.py, the following arguments are required:

- annot_fname : annotation file of one or several of the organisms as input to train and test the model on
- ont : which branch of the gene ontology to test on (acceptable values are 'molecular_function', 'cellular_component', or 'biological_process')
- model_name : name of the model. Result files will be saved with this name prepended.
- network_folder : path of the directory in which both folders 'network_files/' and 'block_matrix_files/' are located
- tax_ids : taxonomy ids for which to load the S matrix and train the autoencoder on. Delimited by commas.
- alpha : Selected alpha value for which the S matrix was constructed.
- test_goid_fname : pickle file containing a list of GO ids for which to train and test the model on.

For example, to run the method for the bacteria with taxonomy IDs 511145,316407,316385,224308,71421,243273 after running the /data/string/pipeline.py script (README in that folder describes usage for that script):

python multispecies.py ../../data/string/string_annot/511145-316407-316385-224308-71421-243273_string.04_2015_annotations.pckl molecular_function bacteria_model_orgs_all_test ../../data/string/ 511145,316407,316385,224308,71421,243273 0.6 ../../data/string/string_annot/511145-316407-316385-224308-71421-243273_molecular_function_train_goids.pckl

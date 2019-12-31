python multispecies.py --tax_ids 511145,316407,316385,224308,71421,243273 --valid_type cv --model_name nn_orig_model --results_path ../results/test/ --data_folder ../../data/string/ --alpha 0.8 --annot ../../data/string/string_annot/511145_string.04_2015_annotations.pckl --ont biological_process --test_goid_fname ../../data/string/string_annot/511145_biological_process_train_goids.pckl --use_orig_features --use_nn_val >> nn_orig_model_biological_process_alpha_0.8_logfile.txt
python multispecies.py --tax_ids 511145,316407,316385,224308,71421,243273 --valid_type cv --model_name nn_orig_model --results_path ../results/test/ --data_folder ../../data/string/ --alpha 0.9 --annot ../../data/string/string_annot/511145_string.04_2015_annotations.pckl --ont biological_process --test_goid_fname ../../data/string/string_annot/511145_biological_process_train_goids.pckl --use_orig_features --use_nn_val >> nn_orig_model_biological_process_alpha_0.9_logfile.txt
python multispecies.py --tax_ids 511145,316407,316385,224308,71421,243273 --valid_type cv --model_name nn_orig_model --results_path ../results/test/ --data_folder ../../data/string/ --alpha 1.0 --annot ../../data/string/string_annot/511145_string.04_2015_annotations.pckl --ont biological_process --test_goid_fname ../../data/string/string_annot/511145_biological_process_train_goids.pckl --use_orig_features --use_nn_val >> nn_orig_model_biological_process_alpha_1.0_logfile.txt

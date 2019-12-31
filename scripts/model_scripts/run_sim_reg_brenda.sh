#!/bin/bash
#module load slurm gcc python3 cuda cudnn/v6.0-cuda-8.0
#source /mnt/home/mbarot/regiona_specTHOR/tensorflow1.5-gpu-env/bin/activate
genv
echo 'Commencing.'
python sim_reg_brenda_multispecies.py /mnt/ceph/users/mbarot/multispecies_deepNF/data/brenda/brenda_string_protein_labels.pckl human_only_sim_reg_brenda /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0

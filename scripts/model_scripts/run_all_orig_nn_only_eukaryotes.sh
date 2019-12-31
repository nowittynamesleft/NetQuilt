module load disBatch

sbatch -N 3 -p gpu  --ntasks-per-node 4 --exclusive --gres=gpu:4 --wrap "disBatch.py -g run_eukaryotes_nn_only_cc.sh"
sbatch -N 3 -p gpu  --ntasks-per-node 4 --exclusive --gres=gpu:4 --wrap "disBatch.py -g run_eukaryotes_nn_only_mf.sh"
sbatch -N 3 -p gpu  --ntasks-per-node 4 --exclusive --gres=gpu:4 --wrap "disBatch.py -g run_eukaryotes_nn_only_bp.sh"

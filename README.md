This is the repository for NetQuilt, a multispecies network-based protein function prediction method incorporating homology information with IsoRank.

In the ./data/string/ directory, there are scripts used to download and preprocess data from STRING used in the method.

In the ./scripts/model_scripts directory, there are scripts used to run the method once IsoRank has been run on all pairs of networks.

    multispecies.py is the main file for running the method. Run python multispecies.py -h for options and use instructions.

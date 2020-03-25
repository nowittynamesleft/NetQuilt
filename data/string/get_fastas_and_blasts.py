import subprocess
import sys
import itertools
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

# Edited for version 10.5 instead of 11



def get_fastas(taxa, fasta_folder='./fasta_files/'):
    pool = Pool(multiprocessing.cpu_count())
    fnames = pool.starmap(get_single_fasta, zip(taxa, itertools.repeat(fasta_folder)))
    #fnames = [pool.map(get_single_fasta, args=(taxon, fasta_folder)) for taxon in taxa]
    # map(pred, arglist) -> arglist = [(a1, a2, a3) ...]
    # map(lambda args: pred(*args)
    '''
    for taxon in taxa:
        #fname = str(taxon) + '.protein.sequences.v10.5.fa.gz'
        #subprocess.run(['wget', '-P', fasta_folder, 'https://version-10-5.string-db.org/download/protein.sequences.v10.5/' + fname])

        fname = str(taxon) + '.protein.sequences.v11.fa.gz'
        #subprocess.run(['wget', '-P', fasta_folder, 'https://string-db.org/download/protein.sequences.v10.5/' + fname])
        subprocess.run(['wget', '-P', fasta_folder, 'https://stringdb-static.org/download/protein.sequences.v11.0/' + fname])
        subprocess.run(['gunzip', '-f', fasta_folder + fname])
        fname = fname[:-3]
        subprocess.run(['makeblastdb', '-in', fasta_folder + fname, '-dbtype', 'prot'])
        fnames.append(fname)
    '''

    return fnames


def get_single_fasta(taxon, fasta_folder):
    fname = str(taxon) + '.protein.sequences.v11.0.fa.gz'
    
    completed = subprocess.run(['wget', '-P', fasta_folder, 'https://stringdb-static.org/download/protein.sequences.v11.0/' + fname])
    if completed.returncode != 0:
        print(taxon + ' download unsuccessful. Check if url ' + 'https://stringdb-static.org/download/protein.sequences.v11.0/' + fname + ' is valid.')
    subprocess.run(['gunzip', '-f', fasta_folder + fname])
    fname = fname[:-3]
    subprocess.run(['makeblastdb', '-in', fasta_folder + fname, '-dbtype', 'prot'])
    return fname

def interspecies_blast(tax_ids, fasta_folder='./fasta_files/', blast_folder='./blast_files/'):
    print('Running BLAST on all combos.')
    combos = list(itertools.combinations_with_replacement(tax_ids, 2))
    print('Number of combos:')
    print(len(combos))
    for combo in combos:
        taxa_1 = combo[0]
        taxa_2 = combo[1]
        #fasta_1 = str(taxa_1) + '.protein.sequences.v10.5.fa'
        #fasta_2 = str(taxa_2) + '.protein.sequences.v10.5.fa'
        fasta_1 = str(taxa_1) + '.protein.sequences.v11.0.fa'
        fasta_2 = str(taxa_2) + '.protein.sequences.v11.0.fa'
        if not Path(fasta_folder + fasta_1).is_file():
            print(str(fasta_1) + ' not found. Downloading it.')
            get_single_fasta(taxa_1, fasta_folder)
        if not Path(fasta_folder + fasta_2).is_file():
            print(str(fasta_2) + ' not found. Downloading it.')
            get_single_fasta(taxa_2, fasta_folder)
        num_cores = multiprocessing.cpu_count()
        print('Num cores: ' + str(num_cores))
        command_list = ['blastp', '-db', fasta_folder + fasta_2, '-query', fasta_folder + fasta_1, '-outfmt', '6', '-evalue', '1e-3', '-out', blast_folder + taxa_1 + '-' + taxa_2 + '_blastp.tab', '-num_threads', str(num_cores)]
        command = ' '.join(command_list)
        print('Running ' + command)
        subprocess.run(command_list)
     

if __name__ == '__main__':
    taxa = sys.argv[1].split(',')
    fasta_fnames = get_fastas(taxa)
    interspecies_blast(fasta_fnames)

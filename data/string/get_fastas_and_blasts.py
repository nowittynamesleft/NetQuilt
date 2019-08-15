import subprocess
import sys
import itertools


def get_fastas(taxa, fasta_folder='./fasta_files/'):
    fnames = []
    for taxon in taxa:
        fname = str(taxon) + '.protein.sequences.v10.5.fa.gz'
        subprocess.run(['wget', 'https://version-10-5.string-db.org/download/protein.sequences.v10.5/' + fname])
        subprocess.run(['mv', fname, fasta_folder])
        subprocess.run(['gunzip', '-f', fasta_folder + fname])
        fname = fname[:-3]
        subprocess.run(['makeblastdb', '-in', fasta_folder + fname, '-dbtype', 'prot'])
        fnames.append(fname)
    return fnames


def interspecies_blast(fasta_fnames, fasta_folder='./fasta_files/', blast_folder='./blast_files/'):
    print('Running BLAST on all combos.')
    combos = list(itertools.combinations(fasta_fnames, 2))
    print('Number of combos:')
    print(len(combos))
    for combo in combos:
        taxa_1 = combo[0].split('.')[0]
        taxa_2 = combo[1].split('.')[0]
        command_list = ['blastp', '-db', fasta_folder + combo[1], '-query', fasta_folder + combo[0], '-outfmt', '6', '-evalue', '1e-3', '-out', blast_folder + taxa_1 + '-' + taxa_2 + '_blastp.tab']
        command = ' '.join(command_list)
        print('Running ' + command)
        subprocess.run(command_list)
     
taxa = sys.argv[1].split(',')
fasta_fnames = get_fastas(taxa)
interspecies_blast(fasta_fnames)

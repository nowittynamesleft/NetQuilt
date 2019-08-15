import sys


def load_mappings(fname_mappings, thresh_identity=95.0):
    # load mapping from 'reviewed_uniprot_2_string' file downloaded from STRING db
    uniprot2string = {}
    fRead = open(fname_mappings, 'rb')
    fRead.readline()
    for line in fRead:
        splitted = line.strip().split('\t')
        uniprot = splitted[1]
        string = splitted[2]
        identity = float(splitted[3])
        uniprot_ac = uniprot.split("|")[0]
        if identity > thresh_identity:
            if uniprot_ac not in uniprot2string:
                uniprot2string[uniprot_ac] = []
            uniprot2string[uniprot_ac].append(string)
    fRead.close()

    return uniprot2string


def write_mappings(uniprot2string, fname_out):
    fWrite = open(fname_out, 'w')
    for uniprot in uniprot2string:
        for string in uniprot2string[uniprot]:
            fWrite.write('%s %s\n' % (uniprot, string))
    fWrite.close()


if __name__ == "__main__":

    uniprot_all_mapping_fname = 'full_uniprot_2_string.04_2015.tsv.gz'
    subprocess.run(['wget', 'https://version-10-5.string-db.org/mapping_files/uniprot_mappings/' + uniprot_all_mapping_fname])
    subprocess.run(['gunzip', '-f', uniprot_all_mapping_fname])

    fname_mapping = tax_id + '_reviewed_uniprot_2_string.04_2015.tsv' 
    fname_out = sys.argv[2]
    uniprot2string = load_mappings(fname_mapping)
    print ("### Number of Uniprot ids mapped to STRING ids: %d" % (len(uniprot2string)))
    write_mappings(uniprot2string, fname_out)

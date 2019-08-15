from Bio import SeqIO
import sys

fasta = sys.argv[1]
outfname = sys.argv[2]
outfile = open(outfname, 'w')

print('Writing to outfile')
for record in SeqIO.parse(fasta, 'fasta'):
    outfile.write(record.id + '\n')
outfile.close()
print('Done')

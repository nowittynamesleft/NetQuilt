import sys
import pandas as pd

# creates taxa\tname\tdomain file from names.dmp and categories.dmp

names_file = sys.argv[1]
categories_file = sys.argv[2]

names_df = pd.read_csv(names_file, header=None, names=['tax_id', 'name_txt', 'unique_name', 'name_class'], sep='\t', index_col=False)
names_df = names_df[names_df.name_class == 'scientific name']
print(names_df)
cats_df = pd.read_csv(categories_file, header=None, names=['domain', 'species-level-taxid', 'tax_id'], sep='\t', index_col=False)
print(cats_df)

joined_df = names_df.merge(cats_df, on='tax_id', how='inner')[['tax_id', 'name_txt', 'domain']].dropna()

print(joined_df)

joined_df.to_csv('taxa_domains.txt', sep='\t', header=True, index=False)

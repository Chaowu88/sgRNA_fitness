'''Prepare the dataset for prediction. The features for the model are: 
seq, guide sequence;
essential, gene essentiality;
ori, guide orientation with respect to genome;
coding, whether the guide targets the coding strand;
pos, guide position.

Gene essentiality is derived from ref. 10.1038/s41467-018-04209-5.
Escherichia coli K-12 genome NC_000913.2 is downloaded from 
https://www.ncbi.nlm.nih.gov/nuccore/49175990.

The position numbering of Bio is different from the ref, and thus is modified to adapt to the ref. 
'''


__author__ = 'Chao Wu'


import pandas as pd
from Bio import SeqIO, motifs


OUT_FILE = 'sgrna_full_length.csv'
GENOME_FILE = 'sequence.gb'
DATA_FILE = '41467_2018_4209_MOESM8_ESM.csv'
GENE_LIST = [
    'ppc', 
    'metE', 
    'ptsH', 
    'cysH', 
    'pfkA', 
    'pfkB', 
    'pykA', 
    'pykF', 
    'pgl', 
    'pgi', 
    'rpe', 
    'gpmA', 
    'gpmM'
]
BEFORE_NTS = 100
AFTER_NTS = 300


def get_gene_essentiality(data_file):

    data = pd.read_csv(data_file, header = 0, index_col = None)
    
    gene_essentiality = data.set_index('gene').to_dict()['essential']

    return gene_essentiality


def extract_gene_sequence(record, gene, strand, start, end, before_nts, after_nts):

    if strand == 1:
        coding_seq = record.seq[start-before_nts:start+after_nts]
        temp_seq = coding_seq.reverse_complement()
    elif strand == -1:
        temp_seq = record.seq[end-after_nts:end+before_nts]
        coding_seq = temp_seq.reverse_complement()
    else:
        raise ValueError(f'{gene} strand unknown')
    
    return coding_seq, temp_seq


def identify_sgrna_targets(motif, search_seq):

    targets = []
    for pos, instance in motif.instances.search(search_seq):
        if pos >= 20:   
            target = search_seq[pos-20:pos+3]
            targets.append(target)
            
    return targets


def main():

    gene_essentiality = get_gene_essentiality(DATA_FILE)

    pam_motif = motifs.create(['AGG', 'TGG', 'CGG', 'GGG'])

    record = SeqIO.read(GENOME_FILE, 'gb')

    model_features = []
    for gene in GENE_LIST:
        essential = gene_essentiality.get(gene, 'NA')

        for feature in record.features:
            if feature.type == 'CDS' and gene in feature.qualifiers['gene']:
                strand = feature.location.strand
                start = feature.location.start
                end = feature.location.end
                
                gene_len = end - start
                #request_len = gene_len if gene_len < AFTER_NTS else AFTER_NTS
                request_len = gene_len

                coding_seq, temp_seq = extract_gene_sequence(
                    record, 
                    gene, 
                    strand, 
                    start, 
                    end, 
                    BEFORE_NTS, 
                    request_len,
                )
                
                coding_targets = identify_sgrna_targets(pam_motif, coding_seq)
                for target_seq in coding_targets:
                    if strand == 1:
                        pos = record.seq.find(target_seq) + 22 
                        ori = '+'
                    elif strand == -1:
                        pos = record.seq.find(target_seq.reverse_complement()) -2 
                        ori = '-'
                    else:
                        raise ValueError(f'{gene} strand unknown')
                    coding = False
                    model_features.append([str(target_seq), essential, ori, coding, pos, gene])
                
                temp_targets = identify_sgrna_targets(pam_motif, temp_seq)
                for target_seq in temp_targets:
                    if strand == 1:
                        pos = record.seq.find(target_seq.reverse_complement()) - 2 
                        ori = '-'
                    elif strand == -1:
                        pos = record.seq.find(target_seq) + 22 
                        ori = '+'
                    else:
                        raise ValueError(f'{gene} strand unknown')
                    coding = True
                    model_features.append([str(target_seq), essential, ori, coding, pos, gene])
                
    model_features = pd.DataFrame(
        model_features, 
        columns = ['seq', 'essential', 'ori', 'coding', 'pos', 'gene']
    )        
    model_features.to_csv(OUT_FILE, header = True, index = False)

                


if __name__ == '__main__':

    main()

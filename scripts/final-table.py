#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path

# VARS
appdir=Path(sys.argv[1])
workdir=Path(sys.argv[2])
resultsdir = workdir/'results'
expr_path = resultsdir/'trans_expr.csv'
gtf_path = workdir/'cuffcompare/cuffcomp.gtf.tracking'
dge_path = resultsdir/'dge.csv'
mlc_path = resultsdir/'ml-classified-new-non-coding.csv'
bed_path = Path(sys.argv[3])

# READ IN THE TABLES AS DATAFRAMES
expr = pd.read_csv(expr_path)
expr = pd.merge(expr['t_name'], expr.filter(axis=1, regex='FPKM'), left_index=True, right_index=True)
expr = pd.merge(expr.iloc[:, [0, -2, -1]], expr.iloc[:, 1:len(expr.columns)-2], left_index=True, right_index=True)

gtf = pd.read_csv(gtf_path, delimiter='\t', header=None)
gtf = pd.merge(gtf, (gtf[4].str.split('|', expand=True)), left_index=True, right_index=True)
gtf.drop(columns=['2_x', '3_x', '4_x', '0_y', '2_y', '3_y', '4_y', 5, 6, 7], inplace=True)
gtf.rename(columns={'0_x': 'transcript', '1_x': 'locus', '1_y': 'transfrag'}, inplace=True)
gtf = gtf[gtf['transfrag'].str.contains('MSTRG')]

dge = pd.read_csv(dge_path)

bed = pd.read_csv(bed_path, delimiter='\t', header=None)
bed.drop(columns=[4, 5], inplace=True)
bed.rename(columns={0:'chr', 1:'start', 2:'end', 3:'name'}, inplace=True)

mlc = pd.read_csv(mlc_path)
mlc.drop(columns=['lncRNA', 'Prob_False'], inplace=True)

final = pd.merge(mlc, bed, on='name')
final = pd.merge(final, final['name'].str.split('.', expand=True), left_index=True, right_index=True)
final.rename(columns={0: 'transcript', 1: 'exon'}, inplace=True)
final.drop(columns='name', inplace=True)

final = pd.merge(final, gtf, on='transcript')

df = final['transfrag'].str.split('.', -1, expand=True)
df['idx'] = df[[0, 1]].agg('.'.join, axis=1)
df = df['idx']
final['idx'] = df
final = pd.merge(final, dge[['geneIDs', 'fc', 'qval']], left_on='idx', right_on='geneIDs', how='left')
final.drop(columns=['idx', 'geneIDs'], inplace=True)

final = pd.merge(final, expr, left_on='transfrag', right_on='t_name', how='left')

final.drop(columns='t_name', inplace=True)

final.fillna(value=0, inplace=True)

# Get the top 20 more probable new lncRNAs
top_20_prob = final.sort_values('Prob_True', ascending=False).iloc[0:20,:]


for i in list(top_20_prob.index):
    coord = list(top_20_prob[['chr', 'start', 'end']].apply(lambda x: '..'.join(x.values.astype(str)), axis=1))
coord = list(map(lambda x: str.replace(x, '..', ':', 1), coord))


# WRITE THE FINAL FILES
with open(resultsdir/'top-20-prob-coord.txt', 'w') as output:
    for element in coord:
        output.write(element + "\n")

top_20_prob.to_csv(resultsdir/'final-results-top-20-prob.csv', index=False)
final.to_csv(resultsdir/'final-results.csv', index=False)

#!/usr/bin/env python3

from cmath import exp
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
cov_path = workdir/'cov'
bed_path = Path(sys.argv[3])

# READ IN THE TABLES AS DATAFRAMES
gtf = pd.read_csv(gtf_path, delimiter='\t', header=None)
gtf = pd.merge(gtf, (gtf[4].str.split('|', expand=True)), left_index=True, right_index=True)
gtf.drop(columns=['2_x', '3_x', '4_x', '0_y', '2_y', '3_y', '4_y', 5, 6, 7], inplace=True)
gtf.rename(columns={'0_x': 'transcript', '1_x': 'locus', '1_y': 'transfrag'}, inplace=True)
gtf = gtf[gtf['transfrag'].str.contains('MSTRG')]

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

if dge_path.is_file():
    dge = pd.read_csv(dge_path)
    df = final['transfrag'].str.split('.', -1, expand=True)
    df['idx'] = df[[0, 1]].agg('.'.join, axis=1)
    df = df['idx']
    final['idx'] = df
    final = pd.merge(final, dge[['geneIDs', 'fc', 'qval']], left_on='idx', right_on='geneIDs', how='left')
    final.drop(columns=['idx', 'geneIDs'], inplace=True)

if expr_path.is_file():
    expr = pd.read_csv(expr_path)
    expr = pd.merge(expr['t_name'], expr.filter(axis=1, regex='FPKM'), left_index=True, right_index=True)
    expr = pd.merge(expr.iloc[:, [0, -2, -1]], expr.iloc[:, 1:len(expr.columns)-2], left_index=True, right_index=True)
    final = pd.merge(final, expr, left_on='transfrag', right_on='t_name', how='left')
    final.drop(columns='t_name', inplace=True)
else:
    cov_dirs = cov_path.glob('**/*')
    cov_files = [x for x in cov_dirs if x.stem == 't_data']
    myVars = vars()
    for i in cov_files:
        myVars[str(i.parent.stem + '_expr')]=pd.read_csv(i, delimiter = '\t')
    expr_tables = [key for key in myVars if '_expr' in key.lower()]
    expr = myVars[expr_tables[0]]['t_name']
    for i in expr_tables:
        expr = pd.merge(expr, myVars[i][['t_name', 'FPKM']], on = 't_name', how = 'outer')
    
    expr.fillna(value = 0, inplace = True)

    for i in range(0,len(expr_tables)):
        mapping = {expr.columns[i+1]: expr_tables[i].split('_')[0] + '_FPKM'}
        expr.rename(columns = mapping, inplace = True)
    expr['mean_FPKM'] = expr.loc[:, expr.columns != 't_name'].mean(axis = 1).round(5)
    filter = ((expr.columns != 't_name') & (expr.columns != 'mean_FPKM'))
    expr['sd_FPKM'] = expr.loc[:, expr.columns == filter].std(axis = 1, numeric_only = float).round(5)



    expr.to_csv(resultsdir/'trans_expr.csv', index=False)

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

#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

# VARS

appdir=Path(sys.argv[1])
workdir=Path(sys.argv[2])
paths_table=workdir/'results/non-coding/features/paths.csv'
outpath=workdir/'results/'

if len(sys.argv) < 4:
    bed_path=workdir/'results/new-non-coding.bed'
else:
    bed_path=Path(sys.argv[3])

prefix=bed_path.stem
prefix=prefix.split('.')[0]

# GET THE TABLES AS PANDAS DFs

readin=pd.read_csv(paths_table, delimiter=',', header=None)
# for index, rows in readin.iterrows():
#     readin.iloc[index, [1]] = appdir + 'lncrna/' + rows[0] + '.tsv'
readin.rename(columns={0: 'track', 1:'url'}, inplace=True)
readin.head(15)


bed=pd.read_csv(bed_path, delimiter='\t', header=None)
bed.columns = ["chr", "start", "end", "name", "score", "strand"]

# Define column names
bw_cols=["name", "size", "covered_bases", "sum", "mean0", "mean", "min", "max"]
bb_cols=["name", "covered_percent", "mean", "min", "max"]


# Create dataframes from extracted features
for i in range(len(readin)):
    if readin.iloc[i,0] == 'GCcont':
        df_gc=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_gc.columns=bw_cols
        df_gc['name'] = df_gc['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'phastCons27':
        df_pc27=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pc27.columns=bw_cols
        df_pc27['name'] = df_pc27['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'phyloP27':
        df_pp27=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pp27.columns=bw_cols
        df_pp27['name'] = df_pp27['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'phyloP124':
        df_pp124=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pp124.columns=bw_cols
        df_pp124['name'] = df_pp124['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'Pol2_S2':
        df_pol2=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pol2.columns=bw_cols
        df_pol2['name'] = df_pol2['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'H3K4me3_S2':
        df_me3=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_me3.columns=bw_cols
        df_me3['name'] = df_me3['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'CAGE_pos_whole_trans':
        df_posTSS_wt=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_posTSS_wt.columns=bw_cols
        df_posTSS_wt['name'] = df_posTSS_wt['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'CAGE_neg_whole_trans':
        df_negTSS_wt=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_negTSS_wt.columns=bw_cols
        df_negTSS_wt['name'] = df_negTSS_wt['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'ReMap':
        df_re=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_re.columns=bb_cols
        df_re['name'] = df_re['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'JASPAR_TF':
        df_tf=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_tf.columns=bb_cols
        df_tf['name'] = df_tf['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'CAGE_pos':
        df_posTSS=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_posTSS.columns=["name", "startPosTSS"]
        df_posTSS['name'] = df_posTSS['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'CAGE_neg':
        df_negTSS=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None, na_values=0)
        df_negTSS.columns=["name", "endNegTSS"]
        df_negTSS['name'] = df_negTSS['name'].map(lambda x: x.rstrip('.'))
    elif readin.iloc[i,0] == 'RNAfold':
        df_2s=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None, na_values=0)
        df_2s.columns=["name", "mfe"]
        df_2s['name'] = df_2s['name'].map(lambda x: x.rstrip('.'))


# Prepare the bed file info
features = bed["name"]
features = pd.merge(features, bed[['name', 'start', 'end']], on=['name'])
features["length"] = features["end"] - features["start"]

# Incorporate best TSS peaks near ('offset' +/- 'start'/'end')
features = pd.merge(features, df_posTSS, on=['name'], how="outer")
features = pd.merge(features, df_negTSS, on=['name'], how="outer")

tss_peak = features[["startPosTSS", "endNegTSS"]]
tss_peak = abs(tss_peak)
tss_peak = tss_peak.max(axis=1)
features["bestTSS"] = tss_peak
features.drop(columns=["startPosTSS", "endNegTSS"], inplace=True)

# Incorporate best TSS peaks inside the whole transcript
features = pd.merge(features, df_posTSS_wt[['name', 'max']], on=['name'], how='outer')
features = pd.merge(features, df_negTSS_wt[['name', 'min']], on=['name'], how='outer')
features.rename(columns={'max': 'PosTSS_inside', 'min': 'NegTSS_inside'}, inplace=True)

tss_peak_ins = features[["PosTSS_inside", "NegTSS_inside"]]
tss_peak_ins = abs(tss_peak_ins)
tss_peak_ins = tss_peak_ins.max(axis=1)
features["bestTSS_inside"] = tss_peak_ins
features.drop(columns=["PosTSS_inside", "NegTSS_inside"], inplace=True)

# Incorporate best metric for each queried bigWig or bigBed files
features = pd.merge(features, df_gc[['name', 'mean']], on=['name'], how='outer')
features.rename(columns={'mean': 'mean_gc'}, inplace=True)

features = pd.merge(features, df_re[['name', 'mean']], on=['name'], how='outer')
features.rename(columns={'mean': 'mean_remap'}, inplace=True)

features = pd.merge(features, df_me3[['name', 'covered_bases']], on=['name'], how='outer')
features.rename(columns={'covered_bases': 'cov_me3'}, inplace=True)
features["cov_me3"] = features["cov_me3"] / features["length"]

features = pd.merge(features, df_tf[['name', 'covered_percent']], on=['name'], how='outer')
features.rename(columns={'covered_percent': 'cov_tfbs'}, inplace=True)

features = pd.merge(features, df_pol2[['name', 'covered_bases']], on=['name'], how='outer')
features.rename(columns={'covered_bases': 'cov_pol2'}, inplace=True)
features["cov_pol2"] = features["cov_pol2"] / features["length"]

features = pd.merge(features, df_pc27[['name', 'mean0']], on=['name'], how='outer')
features.rename(columns={'mean0': 'mean_pcons27'}, inplace=True)

features = pd.merge(features, df_pp27[['name', 'mean0']], on=['name'], how='outer')
features.rename(columns={'mean0': 'mean_pPcons27'}, inplace=True)

features = pd.merge(features, df_pp124[['name', 'mean0']], on=['name'], how='outer')
features.rename(columns={'mean0': 'mean_pPcons124'}, inplace=True)

# Fill NaN with 0
final_features=features.fillna(0)

# Drop rows with no information at all
final_features = final_features.loc[
    final_features['bestTSS'] + 
    final_features['bestTSS_inside'] + 
    final_features['mean_gc'] + 
    final_features['mean_remap'] + 
    final_features['cov_me3'] + 
    final_features['cov_tfbs'] + 
    final_features['cov_pol2'] + 
    final_features['mean_pcons27'] + 
    final_features['mean_pPcons27'] + 
    final_features['mean_pPcons124'] != 0]
# final_features = final_features[final_features['length'] != 0]

final_features.to_csv(outpath/str(prefix+'.csv'), index=False)

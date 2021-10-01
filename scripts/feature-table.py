#!/bin/python3

import sys
import pandas as pd
import numpy as np

paths_table=(sys.argv[1])
bed_path=(sys.argv[2])
outpath=(sys.argv[3])

readin=pd.read_csv(paths_table, delimiter='\t', header=None)

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
    elif readin.iloc[i,0] == 'phastCons27':
        df_pc27=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pc27.columns=bw_cols
    elif readin.iloc[i,0] == 'phyloP27':
        df_pp27=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pp27.columns=bw_cols
    elif readin.iloc[i,0] == 'phyloP124':
        df_pp124=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pp124.columns=bw_cols
    elif readin.iloc[i,0] == 'Pol2_S2':
        df_pol2=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_pol2.columns=bw_cols
    elif readin.iloc[i,0] == 'H3K4me3_S2':
        df_me3=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_me3.columns=bw_cols
    elif readin.iloc[i,0] == 'CAGE_pos_whole_trans':
        df_posTSS_wt=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_posTSS_wt.columns=bw_cols
    elif readin.iloc[i,0] == 'CAGE_neg_whole_trans':
        df_negTSS_wt=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_negTSS_wt.columns=bw_cols
    elif readin.iloc[i,0] == 'ReMap':
        df_re=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_re.columns=bb_cols
    elif readin.iloc[i,0] == 'JASPAR_TF':
        df_tf=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_tf.columns=bb_cols
    elif readin.iloc[i,0] == 'CAGE_pos':
        df_posTSS=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None)
        df_posTSS.columns=["name", "startPosTSS"]
    elif readin.iloc[i,0] == 'CAGE_neg':
        df_negTSS=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None, na_values=0)
        df_negTSS.columns=["name", "endNegTSS"]
    elif readin.iloc[i,0] == 'RNAfold':
        df_2s=pd.read_csv(readin.iloc[i,1], delimiter='\t', header=None, na_values=0)
        df_2s.columns=["name", "mfe"]

# Function to append values of df with different row numbers
def append_columns(ref_df, query_df, ref_col, append_col, **kwargs):
    ref_col = str(ref_col)
    append_col = str(append_col)
    new_col_name = kwargs.get('new_col_name', None)    
    try: new_col_name
    except NameError: new_col_name = None
    if new_col_name is None:
        new_col_name = append_col
    
    df=pd.DataFrame({})
    for name, group in query_df.groupby(ref_col):
        buffer_df = pd.DataFrame({ref_col: group[ref_col][:1]})
        i = 0
        for index, value in group[append_col].iteritems():
            i += 1
            string = new_col_name
            buffer_df[string] = value
        df = df.append(buffer_df)
    features = pd.merge(ref_df, df, how='left', on=ref_col)
    return features

def append_all_cols(ref_df, query_df, ref_col):
    ref_col = str(ref_col)
    for i in range(len(query_df.columns)):
        if query_df.columns[i] == ref_col:
            ref_col = query_df.columns[i]
            for i in range(len(query_df.columns)):
                if query_df.columns[i] != ref_col:
                    append_col = query_df.columns[i]
                    features = append_columns(ref_df, query_df, ref_col, append_col)
                    ref_df = features
    return features

## Create the features table

features = bed["name"]
features = append_columns(features, bed, 'name', 'start')
features = append_columns(features, bed, 'name', 'end')
features["length"] = features["end"] - features["start"]

# Incorporate TSS peaks near ('ofsset' +/- 'start'/'end')
features = append_all_cols(features, df_posTSS, 'name')
features = append_all_cols(features, df_negTSS, 'name')

# Find bes TSS from the positive and negative TSS values arround 5' and 3' of transcript
tss_peak = features[["startPosTSS", "endNegTSS"]]
tss_peak = abs(tss_peak)
tss_peak = tss_peak.max(axis=1)
features["bestTSS"] = tss_peak
features.drop(columns=["startPosTSS", "endNegTSS"], inplace=True)

# Incorporate TSS peaks inside the transcript
features = append_columns(features, df_posTSS_wt, 'name', 'max', new_col_name='PosTSS_inside')
features = append_columns(features, df_negTSS_wt, 'name', 'min', new_col_name='NegTSS_inside')

# Find best TSS inside the transcript itself
tss_peak_ins = features[["PosTSS_inside", "NegTSS_inside"]]
tss_peak_ins = abs(tss_peak_ins)
tss_peak_ins = tss_peak_ins.max(axis=1)
features["bestTSS_inside"] = tss_peak_ins
features.drop(columns=["PosTSS_inside", "NegTSS_inside"], inplace=True)

# Incorporate best metric for each queried bigWig or bigBed files
features = append_columns(features, df_gc, 'name', 'mean', new_col_name='mean_gc')
features = append_columns(features, df_re, 'name', 'mean', new_col_name='mean_remap')
features = append_columns(features, df_me3, 'name', 'covered_bases', new_col_name='cov_me3')
features["cov_me3"] = features["cov_me3"] / features["length"]
features = append_columns(features, df_tf, 'name', 'covered_percent', new_col_name='cov_tfbs')
features = append_columns(features, df_pol2, 'name', 'covered_bases', new_col_name='cov_pol2')
features["cov_pol2"] = features["cov_pol2"] / features["length"]
features = append_columns(features, df_pc27, 'name', 'mean0', new_col_name='mean_pcons27')
features = append_columns(features, df_pp27, 'name', 'mean0', new_col_name='mean_pPcons27')
features = append_columns(features, df_pp124, 'name', 'mean0', new_col_name='mean_pPcons124')

features.dropna(axis=0, how='any', inplace=True)

features.to_csv(outpath + '/X_train_lncrna.csv')
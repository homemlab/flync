#!/usr/bin/env python3

import sys
import pickle
import pandas as pd
from pathlib import Path

# VARS
appdir=Path(sys.argv[1])
workdir=Path(sys.argv[2])
outpath=workdir/'results'
model_path=appdir/'model'/'rf_dm6_lncrna_classifier.model'

if len(sys.argv) < 4:
    predict_table=outpath/'new-non-coding.csv'
    prefix=predict_table.stem
    prefix=prefix.split('.')[0]
else:
    try:
        prefix=sys.argv[3]
        predict_table=outpath/str(prefix+'.csv')
    except (IndexError, FileNotFoundError) as e:
        print(f"Couldn't find the correct table: {e}")
        sys.exit(1)

# READ THE FEATURE TABLE FOR PREDICTION
pt=pd.read_csv(predict_table)

# KEEP ONLY REQUIRED FEATURES
# Both these features were removed from the final model for simplicity
# Since they were highly correlated with other features and adding no value to the prediction
pt.drop('cov_me3', axis=1, inplace=True)
pt.drop('mean_pPcons27', axis=1, inplace=True)

model_pt=pt.loc[:, 'length':'mean_pPcons124']

# LOAD MODEL
model=pickle.load(open(model_path, 'rb'))

# PREDICT
result=pd.DataFrame(model.predict(model_pt))
result.rename(columns={0:'lncRNA'}, inplace=True)

result_prob=pd.DataFrame(model.predict_proba(model_pt))
result_prob.rename(columns={0:'Prob_False', 1:'Prob_True'}, inplace=True)

final_result=pd.concat([pt['name'], result, result_prob['Prob_False'], result_prob['Prob_True']], axis=1, verify_integrity=True)

pos_final_result=final_result[final_result['lncRNA'] == True]

# WRITE TABLES TO FILE
final_result.to_csv(outpath/str('all-ml-class-'+prefix+'.csv'), index=False, header=True)

pos_final_result['name'].to_csv(outpath/str('ml-classified-names-'+prefix+'.csv'), index=False, header=False)
pos_final_result.to_csv(outpath/str('ml-classified-'+prefix+'.csv'), index=False, header=True)

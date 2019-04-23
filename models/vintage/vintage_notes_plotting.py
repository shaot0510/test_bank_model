import os
import importlib
import math
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

BASE_PATH = os.path.join(os.getcwd(), 'data')
DIR_PATH = os.path.join(BASE_PATH, 'fannie_mae_data')
DATA_PATH = os.path.join(DIR_PATH, 'clean')
EXPORT_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'results')
ECON_PATH = os.path.join(BASE_PATH, 'economic')
FILENAME = '10000.2010Q1.csv'

############################################################
# HELPER FXNS
############################################################
def plot_path(fname):
    return os.path.join(EXPORT_PATH, fname)

############################################################
# READ/PROCESS/CLEAN DATA
############################################################
df = pd.read_csv(os.path.join(DATA_PATH, FILENAME),
                 engine='c',
                 parse_dates=['PRD', 'ORIG_DTE'])

# attach econ vars
df_econ = pd.read_csv(os.path.join(ECON_PATH, 'agg_ntnl_mnthly.csv'),
                      parse_dates=['DATE'])
df = df.merge(df_econ, how='left', left_on='PRD',
              right_on='DATE', copy=False)

# delete unnecessary variables
del df['DATE'], df['Unnamed: 0']

# change date format
df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')

# create age and other vars
df['AGE'] = (df['PRD'] - df['ORIG_DTE']).astype(int)

############################################################
# Summary
############################################################
pct_dflt = df.groupby('LOAN_ID')['DID_DFLT'].last().mean()

############################################################
# PLOTTING
############################################################
age_by_dflt = df.groupby('AGE')['DID_DFLT'].sum()

x, y = age_by_dflt.index, age_by_dflt.values

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(x=x, y=y, s=1, c='#F8C471')
ax.set_xlabel('AGE')
ax.set_ylabel('dflt_pct')
ax.legend()
ax.set_ylim(-0.0003, None)
fig.suptitle('# Default over Age')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(plot_path('dflt_age_per_loan.png'))
plt.close('all')


# df['dflt_pct'] = df['DFLT_AMT'] / df['ORIG_AMT_sum']
# df['min_dflt'] = 1*(df['dflt_pct'] > 0)
# df['net_loss_pct'] = 0
# df.loc[df['min_dflt'] == 1,
#        'net_loss_pct'] = df['NET_LOSS_AMT'] / df['DFLT_AMT']

# # what we ultiamtely want to predict
# df['final_loss_pct'] = df['NET_LOSS_AMT'] / df['ORIG_AMT_sum']

# # # create other variables
# # regex = re.compile('_wv$')
# # wghted_cols = [regex.sub('', c) for c in df.columns if regex.search(c)]
# # for col in wghted_cols:
# #     df['{0}_cv'.format(col)] = np.sqrt(
# #         df['{0}_wv'.format(col)])/df['{0}_wm'.format(col)]

# # drop columns with too many NA values
# df = utility.drop_NA_cols(df)

# # remove all NA rows
# a = df.shape
# df.dropna(axis=0, how='any', inplace=True)
# print('Reduced rows from {0} -> {1}'.format(a, df.shape))

# # isolate covariates
# regex = re.compile('^MI_TYPE|^MI_PCT')
# all_vars = [
#     v for v in df.columns if v not in MAIN_VARS and not regex.search(v)]


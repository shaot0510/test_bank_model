#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:57:14 2019

@author: tongshao
"""

import os
import argparse
import re
import sys
import glob
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
sys.path.insert(0, ROOT_PATH)

from modules import models, bootstrap, utility


PLOT_PARAMS = {'test': {'s': 4, 'color': '#c04e01', 'alpha': 0.7},
               'train': {'s': 2, 'color': '#ffa756', 'alpha': 0.4},
               'actual line': {'lw': 2, 'color': '#0165fc',
                               'linestyle': '--'},
               'test roc curve': {'color': '#c04e01', 'lw': 2},
               'train roc curve': {'color': '#ffa756', 'lw': 2}}
                                   
def sorted_dir(folder):
    def getmtime(name):
        path = os.path.join(folder, name)
        return os.path.getmtime(path)

    return sorted(os.listdir(folder), key=getmtime, reverse=True)
                                   
BASE_PATH = os.path.join(ROOT_PATH, 'data')
DIR_PATH = os.path.join(BASE_PATH, 'fannie_mae_data')
DATA_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'data')
SAVED_PATH = os.path.join(DATA_PATH, 'saved_model')
first_file = sorted(os.listdir(SAVED_PATH))[0]


def main(args):
    
    modelname = args.modelname
    dataname = args.dataname
    
    FILE_PATH = os.path.join(DATA_PATH, dataname)
    EXPORT_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'results')
    ECON_PATH = os.path.join(BASE_PATH, 'economic')
    ECON_FILE = os.path.join(ECON_PATH, 'eco_df/*.csv')


    
    if not os.path.exists(EXPORT_PATH):
        print('\nCreating results directory...')
        os.makedirs(EXPORT_PATH)
    output_dir_name = modelname + '-results' + datetime.datetime.now().strftime("%y%m%d%H%M%S")
    output_dir_path = os.path.join(EXPORT_PATH, output_dir_name)
    if not os.path.exists(output_dir_path):
        print('\nCreating directory at export location...')
        os.makedirs(output_dir_path)
        
    model1, model2, model3 = utility.load_all(modelname)

    df_raw = pd.read_csv(FILE_PATH, parse_dates=['PRD', 'ORIG_DTE'])
    
    
    eco_list = glob.glob(ECON_FILE)
    df_econ_all = []
    n = len(eco_list)
    for i in range(n):
        file = pd.read_csv(eco_list[i], parse_dates = ['DATE'])
        df_econ_all.append(file)
    
    for date, dall in df_raw.groupby('ORIG_DTE'):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel('Age of loan')
        ax.set_ylabel('% of ORIG_AMT lost')
        loan_count = dall['LOAN_ID_count'].values[0]
        ax.set_title('Projected Loss: {0} loans from {1}'.format(loan_count, date))
        index = 1
        for df_econ in df_econ_all:
            df = dall.merge(df_econ, how='left', left_on='PRD',right_on='DATE', copy=False)
            del df['DATE'], df['Unnamed: 0']
    
            # change date format
            df.loc[:, 'PRD'] = df.loc[:, 'PRD'].dt.to_period('m')
            df.loc[:, 'ORIG_DTE'] = df.loc[:, 'ORIG_DTE'].dt.to_period('m')
        
            # create age and other vars
            df['AGE'] = (df['PRD'] - df['ORIG_DTE']).astype(int)
            df['dflt_pct'] = df['DFLT_AMT'] / df['ORIG_AMT_sum']
            df['did_dflt'] = 1*(df['dflt_pct'] > 0)
            df['net_loss_pct'] = 0
            df.loc[df['did_dflt'] == 1,
                   'net_loss_pct'] = df['NET_LOSS_AMT'] / df['DFLT_AMT']
        
            # what we ultiamtely want to predict
            df['dflt_loss_pct'] = df['NET_LOSS_AMT'] / df['ORIG_AMT_sum']
            
            
            # create other variables
            regex = re.compile('_wv$')
            wghted_cols = [regex.sub('', c) for c in df.columns if regex.search(c)]
            for col in wghted_cols:
                df['{0}_cv'.format(col)] = np.sqrt(
                    df['{0}_wv'.format(col)])/df['{0}_wm'.format(col)]
        
            # drop columns with too many NA values
            df = utility.drop_NA_cols(df)
        
            # remove all NA rows
            df.dropna(axis=0, how='any', inplace=True)
            
            test = df.copy()
            
            # stage 1
            test['PD_pred'] = model1.make_pred(pred_input=df)
        
            # stage 2
            test['EAD_pred'] = np.exp(model2.make_pred(pred_input=df))
        
            # stage 3
            test['LGD_pred'] = model3.make_pred(pred_input=df)
        
            test['L_pred'] = test['PD_pred'] * test['EAD_pred'] * test['LGD_pred']
        
        
            df_test = test[['AGE', 'L_pred']]  
            
            
            cumsum = df_test.set_index('AGE').cumsum()
            sns.lineplot(x='AGE', y='L_pred', data=cumsum.reset_index(),
                     ax=ax, linewidth=1.5, label='exo_data -' + str(index))
            ax.legend(loc="lower right")
            index += 1
            
        plt.tight_layout()
        plotname = '{2}.{1}.{0}.png'.format(date, modelname, dataname)
        plt.savefig(os.path.join(output_dir_path, plotname), dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf() 
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load the given model, test the new data and outputs prediction plots.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataname', type=str, help='name of data folder', default='vintage_filelist_50.csv')
    parser.add_argument('-m','--modelname', type=str, help='index and the data of models', default=first_file)
    args = parser.parse_args()
    main(args)        
    
    
    
    
import os
import argparse
import re
import datetime
import importlib
import sys

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
first_file = sorted_dir(SAVED_PATH)[0]



def main(args):
    
    modelname = args.modelname
    dataname = args.dataname
    
    FILE_PATH = os.path.join(DATA_PATH, dataname)
    EXPORT_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'results')
    ECON_PATH = os.path.join(BASE_PATH, 'economic')
    
    if not os.path.exists(EXPORT_PATH):
        print('\nCreating results directory...')
        os.makedirs(EXPORT_PATH)
    output_dir_name = modelname + '-results'
    output_dir_path = os.path.join(EXPORT_PATH, output_dir_name)
    if not os.path.exists(output_dir_path):
        print('\nCreating directory at export location...')
        os.makedirs(output_dir_path)
        
    model1, model2, model3 = utility.load_all(modelname)

    train = model1.train
    df = pd.read_csv(FILE_PATH)
    test = df.copy()
    
    # stage 1
    train['PD_pred'] = model1.make_pred(use_train=True)
    test['PD_pred'] = model1.make_pred(pred_input=df)

    # stage 2
    train['EAD_pred'] = np.exp(model2.make_pred(pred_input=train))
    test['EAD_pred'] = np.exp(model2.make_pred(pred_input=df))

    # stage 3
    train['LGD_pred'] = model3.make_pred(pred_input=train)
    test['LGD_pred'] = model3.make_pred(pred_input=df)



    train['L_pred'] = train['PD_pred'] * \
    train['EAD_pred'] * train['LGD_pred']
    test['L_pred'] = test['PD_pred'] * test['EAD_pred'] * test['LGD_pred']


    df_all = test[['ORIG_DTE', 'AGE', 'dflt_loss_pct', 'L_pred']]
    df_stage1 = test[['ORIG_DTE', 'AGE', 'did_dflt', 'PD_pred']]
    df_stage2 = test[['ORIG_DTE', 'AGE', 'dflt_pct', 'EAD_pred']]
    df_stage3 = test[['ORIG_DTE', 'AGE', 'net_loss_pct', 'LGD_pred']]
    
    for date, dall in df_all.groupby('ORIG_DTE'):
        print('\nGenerating plots for {0}...'.format(date))
        ds = []
        for df_ in [df_stage1, df_stage2, df_stage3]:
            to_append = df_[df_['ORIG_DTE'] == date]
            del to_append['ORIG_DTE']
            ds.append(to_append)
        del dall['ORIG_DTE']
        
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # stage 1 roc
        ax, d1 = axes[0, 0], ds[0]
        fpr, tpr = utility.get_roc_curve(train['did_dflt'], train['PD_pred'])
        roc_auc = utility.get_auc(fpr, tpr)
        ax.plot(fpr, tpr, label='Train AUC = {0:.2f}'.format(roc_auc), **PLOT_PARAMS['train roc curve'])

        fpr, tpr = utility.get_roc_curve(d1['did_dflt'], d1['PD_pred'])
        roc_auc = utility.get_auc(fpr, tpr)
        ax.plot(fpr, tpr, label='Test AUC = {0:.2f}'.format(roc_auc), **PLOT_PARAMS['test roc curve'])

        ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Probability of Default: ROC curve')
        ax.legend(loc="lower right")

        # stage 2 predicted vs. actual
        ax, d2 = axes[0, 1], ds[1]

        ax.plot([-0.5, 1.5], [-0.5, 1.5], **PLOT_PARAMS['actual line'])
        c, d = train['EAD_pred'], train['dflt_pct']
        ax.scatter(c, d, label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()), **PLOT_PARAMS['train'])
        a, b = d2['EAD_pred'], d2['dflt_pct']
        ax.scatter(a, b, label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()), **PLOT_PARAMS['test'])
        xymax = min(0.5, max(a.max(), b.max()) * 1.05)
        xymin = max(-0.01, min(a.min(), b.min()) * 0.95)
        ax.set_xlim([xymin, xymax])
        ax.set_ylim([xymin, xymax])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Exposure at Default: % balance at default')
        ax.legend(loc="lower right")



        # stage 3 predicted vs. actual
        ax, d3 = axes[1, 0], ds[2]

        ax.plot([-10, 10], [-10, 10], **PLOT_PARAMS['actual line'])
        c, d = train['LGD_pred'], train['net_loss_pct']
        ax.scatter(c, d, label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()), **PLOT_PARAMS['train'])
        a, b = d3['LGD_pred'], d3['net_loss_pct']
        ax.scatter(a, b, label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()), **PLOT_PARAMS['test'])

        xymax = min(2, max(a.max(), b.max()) * 1.05)
        xymin = max(-0.5, min(a.min(), b.min()) * 0.95)
        ax.set_xlim([xymin, xymax])
        ax.set_ylim([xymin, xymax])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Loss given Default: % lost at default')
        ax.legend(loc="lower right")
        

        

        # plot settings
        plt.tight_layout()
        plotname = '{2}.{1}.{0}.png'.format(date, modelname, dataname)
        plt.savefig(os.path.join(output_dir_path, plotname), dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load the given model, test the new data and outputs prediction plots.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataname', type=str, help='name of data folder', default='df_test1.csv')
    parser.add_argument('-m','--modelname', type=str, help='index and the data of models', default=first_file)
    args = parser.parse_args()
    main(args)        
    
    
    
    
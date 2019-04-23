import os
import argparse
import re
import datetime
import importlib

import numpy as np
from numpy import polyfit, poly1d
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d, BSpline, CubicSpline, splrep, splev
from modules import models, bootstrap, utility

# remove settingcopywithwarning
pd.options.mode.chained_assignment = None

# parameters for plotting
PLOT_PARAMS = {'test': {'s': 4, 'color': '#c04e01', 'alpha': 0.7},
               'train': {'s': 2, 'color': '#ffa756', 'alpha': 0.4},
               'actual line': {'lw': 2, 'color': '#0165fc',
                               'linestyle': '--'},
               'test roc curve': {'color': '#c04e01', 'lw': 2},
               'train roc curve': {'color': '#ffa756', 'lw': 2}}

# list of main variables
MAIN_VARS = ['ORIG_DTE', 'PRD', 'DFLT_AMT', 'NET_LOSS_AMT', 'dflt_pct',
             'min_dflt', 'final_loss_pct']
    
def get_fitted_model(train, test, choice):
    # print('\nFitting {0} with formula {1} and kwargs {2}'
    #       .format(choice[0], choice[1], choice[2]))
    model_class_ = getattr(models, choice[0])
    model = model_class_(train, test, formula=choice[1])
    model.fit_model(model_kwargs=choice[2])
    return model


def fit_stage1(model_spec, train, test,
               pred_train=False, pred_input_data=None,
               return_model=False):
    model = get_fitted_model(train, test, model_spec)
    test_pred = model.make_pred()
    if pred_train:
        if pred_input_data is None:
            train_pred = model.make_pred(use_train=True)
        else:
            train_pred = model.make_pred(pred_input=pred_input_data)
    else:
        train_pred = None

    if return_model:
        return train_pred, test_pred, model
    else:
        return train_pred, test_pred


def fit_stage2(model_spec, train, test, pred_train=False,
               pred_input_data=None, return_model=False):
    model = get_fitted_model(train, test, model_spec)
    test_pred = np.exp(model.make_pred())    
    if pred_train:
        if pred_input_data is None:
            train_pred = np.exp(model.make_pred(use_train=True))
        else:
            train_pred = np.exp(model.make_pred(pred_input=pred_input_data))
    else:
        train_pred = None

    if return_model:
        return train_pred, test_pred, model
    else:
        return train_pred, test_pred



def fit_stage3(model_spec, train, test, pred_train=False,
               pred_input_data=None, return_model=False):
    model = get_fitted_model(train, test, model_spec)    
    test_pred = model.make_pred()
    if pred_train:
        if pred_input_data is None:
            train_pred = model.make_pred(use_train=True)
        else:
            train_pred = model.make_pred(pred_input=pred_input_data)
    else:
        train_pred = None

    if return_model:
        return train_pred, test_pred, model
    else:
        return train_pred, test_pred



def fit_3stages(model_specs, train, test, pred_train=False):
    # stage 1
    train_PD_pred, test_PD_pred = fit_stage1(model_specs[0], train, test,
                                               pred_train)
    # stage 2
    train_pos = train[train['dflt_pct'] > 0]
    train_EAD_pred, test_EAD_pred = fit_stage2(model_specs[1], train_pos, test,
                                                  pred_train, train)
    # stage 3
    train_LGD_pred, test_LGD_pred = fit_stage3(model_specs[2], train_pos, test,
                                                  pred_train, train)
    if pred_train:
        return (train_PD_pred * train_EAD_pred * train_LGD_pred,
                test_PD_pred * test_EAD_pred * test_LGD_pred)
    else:
        return (None, test_PD_pred * test_EAD_pred * test_LGD_pred)


def main(args):
    ############################################################
    # PATHS
    ############################################################
    BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../../data')
    # BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                     '../../data')

    DIR_PATH = os.path.join(BASE_PATH, args.dataname)
    DATA_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'data')
    EXPORT_PATH = os.path.join(DIR_PATH, 'vintage_analysis', 'results')
    ECON_PATH = os.path.join(BASE_PATH, 'economic')
    FILENAME = args.filename

    ############################################################
    # HELPER FXNS
    ############################################################
    def plot_path(fname):
        return os.path.join(EXPORT_PATH, fname)

    ############################################################
    # READ/PROCESS/CLEAN DATA
    ############################################################
    df = pd.read_csv(os.path.join(DATA_PATH, FILENAME),
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
    df['dflt_pct'] = df['DFLT_AMT'] / df['ORIG_AMT_sum']
    df['min_dflt'] = 1*(df['dflt_pct'] > 0)
    df['net_loss_pct'] = 0
    df.loc[df['min_dflt'] == 1,
           'net_loss_pct'] = df['NET_LOSS_AMT'] / df['DFLT_AMT']

    # what we ultiamtely want to predict
    df['final_loss_pct'] = df['NET_LOSS_AMT'] / df['ORIG_AMT_sum']

    # # create other variables
    # regex = re.compile('_wv$')
    # wghted_cols = [regex.sub('', c) for c in df.columns if regex.search(c)]
    # for col in wghted_cols:
    #     df['{0}_cv'.format(col)] = np.sqrt(
    #         df['{0}_wv'.format(col)])/df['{0}_wm'.format(col)]

    # drop columns with too many NA values
    df = utility.drop_NA_cols(df)

    # remove all NA rows
    a = df.shape
    df.dropna(axis=0, how='any', inplace=True)
    print('Reduced rows from {0} -> {1}'.format(a, df.shape))

    # isolate covariates
    regex = re.compile('^MI_TYPE|^MI_PCT')
    all_vars = [
        v for v in df.columns if v not in MAIN_VARS and not regex.search(v)]

    ############################################################
    # PLOTTING
    ############################################################
    yr2grp = {'00-01': [2000, 2001],
              '02-05': [2002, 2003, 2004, 2005],
              '06-08': [2006, 2007, 2008],
              '09-10': [2009, 2010]}
    grp2yr = {yr: k for k, v in yr2grp.items() for yr in v}
    
    # categorize years by yr2grp
    df['ORIG_YR'] = df.ORIG_DTE.apply(lambda x: x.year)
    df['ORIG_YR_GRP'] = df.ORIG_YR.apply(lambda yr: grp2yr[yr] if yr in grp2yr
                                         else 'OTHER')
    df_pos = df[df['min_dflt'] == 1]
    
    #################### ag vs. dflt ####################
    to_plot = df[['AGE', 'dflt_pct']]

    # find median loss per age
    meds = to_plot.groupby('AGE').median()
    x_uniq, y_median = meds.index, meds.values

    # fit bspline of deg = 3
    sfit = splrep(x=meds.index, y=meds,
                  k=3, t=np.quantile(x_uniq, [0.33,0.66]))
    
    x, y = to_plot['AGE'], to_plot['dflt_pct']
    all_x = np.arange(x.min(), x.max() + 1)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(x=x.values, y=y.values, s=1, c='#F8C471')
    ax.plot(all_x, splev(all_x, sfit), c='r', lw=3,
            label='B-Spline Fit')
    ax.set_xlabel('AGE')
    ax.set_ylabel('dflt_pct')
    ax.legend()
    ax.set_ylim(-0.0003, None)
    fig.suptitle('% Default over the Age of Vintage')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('dflt_age.png'))
    plt.close('all')

    #################### age vs. dflt by year ####################
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(x=df['AGE'].values, y=df['dflt_pct'].values, s=1, c='#F8C471')
    
    for orig_yr, df_yr in df.groupby('ORIG_YR'):
        meds = df_yr.groupby('AGE')['dflt_pct'].median()
        x_uniq, y_median = meds.index, meds.values
        sfit = splrep(x=x_uniq, y=y_median,
                      k=3, t=np.quantile(x_uniq, [0.33,0.66]))
        
        x, y = df_yr['AGE'], df_yr['dflt_pct']
        all_x = np.arange(x.min(), x.max() + 1)

        ax.plot(all_x, splev(all_x, sfit), lw=2,
                label=str(orig_yr))
    ax.set_xlabel('AGE')
    ax.set_ylabel('dflt_pct')
    ax.set_ylim(-0.0003, None)
    ax.legend()
    fig.suptitle('% Default over Age of Vintage by Year')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('dflt_age_by_year.png'))
    plt.close('all')

    #################### age vs. dflt by year up to < 2007 ####################
    df_prerec = df[df.PRD < pd.Period('2007-01', freq='m')]
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(x=df_prerec['AGE'].values, y=df_prerec['dflt_pct'].values, s=1,
               c='#F8C471')
    
    for orig_yr, df_yr in df_prerec.groupby('ORIG_YR'):
        meds = df_yr.groupby('AGE')['dflt_pct'].median()
        x_uniq, y_median = meds.index, meds.values
        sfit = splrep(x=x_uniq, y=y_median,
                      k=3, t=np.quantile(x_uniq, [0.33,0.66]))
        
        x, y = df_yr['AGE'], df_yr['dflt_pct']
        all_x = np.arange(x.min(), x.max() + 1)

        ax.plot(all_x, splev(all_x, sfit), lw=2,
                label=str(orig_yr))

    ax.set_xlabel('AGE')
    ax.set_ylabel('dflt_pct')
    ax.set_ylim(-0.0003, None)
    ax.legend()
    fig.suptitle('% Default over the Age of Vintage pre-recession')
    fig.tight_layout(rect=[0, 0.03,1, 0.95])
    fig.savefig(plot_path('dflt_age_by_year_prerec.png'))
    plt.close('all')

    #################### age vs. dflt by year post rec ####################
    df_postrec = df[df.ORIG_YR >= 2009]
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(x=df_postrec['AGE'].values, y=df_postrec['dflt_pct'].values, s=1,
               c='#F8C471')
        
    for orig_yr, df_yr in df_postrec.groupby('ORIG_YR'):
        meds = df_yr.groupby('AGE')['dflt_pct'].median()
        x_uniq, y_median = meds.index, meds.values
        sfit = splrep(x=x_uniq, y=y_median,
                      k=3, t=np.quantile(x_uniq, [0.33,0.66]))

        x, y = df_yr['AGE'], df_yr['dflt_pct']
        all_x = np.arange(x.min(), x.max() + 1)

        ax.plot(all_x, splev(all_x, sfit), lw=2,
                label=str(orig_yr))

    ax.set_xlabel('AGE')
    ax.set_ylabel('dflt_pct')
    ax.set_ylim(-0.0003, None)
    ax.legend()
    fig.suptitle('% Default over the Age of Vintage post-recession')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('dflt_age_by_year_postrec.png'))
    plt.close('all')

    # the dflt curve does peak early and slope down
    # but hard to untangle age effects from recession/macro effects
    
    ############################################################
    # dflt_pct
    ############################################################
    #################### plot distribtiosn ####################
    to_hist, yr_grps = [], []
    for orig_yr_grp, df_yr in df.groupby('ORIG_YR_GRP'):
        data = df_yr['dflt_pct']
        to_hist.append(data)
        yr_grps.append('{0}: {1:.1f}% = no dflt'
                       .format(orig_yr_grp,
                               100 * (data==0).mean()))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(to_hist, label=yr_grps,
            # normalize the histograms
            density=True)

    ax.set_xlabel('dflt_pct')
    ax.legend(loc='best')
    fig.suptitle('Distribution of % Default per Group')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('dflt_pct dist by grp.png'))
    plt.close('all')

    #################### plot relationships ####################
    to_plot = df[['dflt_pct', 'CSCORE_MN_wm', 'DTI_wm', 'LOAN_ID_count',
                  'ORIG_YR_GRP']]
    sns.pairplot(to_plot, hue='ORIG_YR_GRP', markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('pairplot of dflt_pct'))

    to_plot = df[['dflt_pct', 'MR', 'rGDP', 'AGE', 'ORIG_YR_GRP']]
    sns.pairplot(to_plot, hue='ORIG_YR_GRP', markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('pairplot of dflt_pct2'))

    ############################################################
    # net_loss_pct
    ############################################################
    #################### distributions ####################
    fig, ax = plt.subplots(figsize=(8, 8))
    for orig_yr_grp, df_yr in df_pos.groupby('ORIG_YR'):
        ax.hist(df_yr['net_loss_pct'], label=str(orig_yr_grp),
                # normalize the histograms
                alpha=0.5,
                density=True)
    ax.set_xlabel('net_loss_pct')
    ax.legend(loc='best')
    fig.suptitle('Distribution: % of Default Amount Lost per Group')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('net_loss_pct dist by yr'))
    plt.close('all')
    

    #################### relationships ####################
    to_plot = df[['net_loss_pct', 'CSCORE_MN_wm', 'DTI_wm', 'LOAN_ID_count',
                  'ORIG_YR_GRP']]
    sns.pairplot(to_plot, hue='ORIG_YR_GRP', markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('pairplot of net_loss_pct'))

    to_plot = df[['net_loss_pct', 'MR', 'rGDP', 'AGE', 'ORIG_YR_GRP']]
    sns.pairplot(to_plot, hue='ORIG_YR_GRP', markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('pairplot of net_loss_pct2'))

    # not as strong of relationship as dflt_pct
    # but still some structue (Ex. MR)

    ############################################################
    # Low default analysis
    ############################################################
    cutoff_date = pd.Period('2009-06-01', freq='m')
    df_low = df[df.ORIG_DTE >= cutoff_date]
    df_high = df[df.ORIG_DTE < cutoff_date]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist([df_low['dflt_pct'],
              df_high['dflt_pct']],
             label=['Low Period', 'High Period'],
            density=True)
    ax.set_xlabel('dflt_pct')
    ax.legend(loc='best')
    fig.suptitle('Distribution: % Default pre- & post- 2009')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path('low_dflt histogram'))
    plt.close('all')
    
    #################### plot relationships with dflt_pct ####################
    to_plot = df_low[['dflt_pct', 'CSCORE_MN_wm', 'DTI_wm', 'LOAN_ID_count',
                      'ORIG_YR_GRP']]
    sns.pairplot(to_plot, markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('low_dflt: pairplot of dflt_pct'))

    to_plot = df[['dflt_pct', 'MR', 'rGDP', 'AGE', 'ORIG_YR_GRP']]
    sns.pairplot(to_plot, markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('low_dflt: pairplot of dflt_pct2'))

    # fortunately, there is still some relationship btw dflt_pct and vars

    #################### plot relationships with net_loss_pct ####################
    to_plot = df_low[['net_loss_pct', 'CSCORE_MN_wm', 'DTI_wm', 'LOAN_ID_count',
                      'ORIG_YR_GRP']]
    sns.pairplot(to_plot, markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('low_dflt: pairplot of net_loss_pct'))

    to_plot = df[['net_loss_pct', 'MR', 'rGDP', 'AGE', 'ORIG_YR_GRP']]
    sns.pairplot(to_plot, markers='.', height=2)
    plt.tight_layout()
    plt.savefig(plot_path('low_dflt: pairplot of net_loss_pct2'))

    
    ############################################################
    # MODEL SPECS
    ############################################################
    # Stage 1
    # selected = ['UNEMP', 'HPI', 'rGDP', 'ORIG_AMT_sum', 'ORIG_CHN_R_wv',
    #             'LOAN_ID_count', 'MR', 'LIBOR', 'DTI_wm', 'AGE', 'CPI',
    #             'ORIG_RT_wm', 'ORIG_CHN_R_wm', 'ORIG_RT_wv',
    #             'DTI_wv', 'OCLTV_wv',
    #             'ORIG_CHN_C_wv',
    #             'CSCORE_MN_wv', 'CSCORE_MN_wm']    
    selected = ['CSCORE_MN_wm', 'CSCORE_MN_wv',
                'DTI_wm', 'DTI_wv', 'LOAN_ID_count',
                'OCLTV_wm', 'OCLTV_wv',
                'ORIG_AMT_sum',
                'ORIG_CHN_C_wm',
                'ORIG_CHN_R_wm',
                'ORIG_RT_wm', 'ORIG_RT_wv',
                'PURPOSE_P_wm', 'PURPOSE_P_wv',
                'PURPOSE_R_wm', 'PURPOSE_R_wv',
                'CPI', 'UNEMP', 'HPI', 'rGDP', 'AGE']
    gbm_formula1 = 'min_dflt ~ -1 + {0}'.format(' + '.join(selected))
    kw1 = {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 0.5,
           'min_impurity_decrease': 0.0001, 'n_estimators': 10, 'subsample': 0.1}

    # Stage 2
    rfr_formula2 = ('np.log(dflt_pct) ~ -1 + {0}'
                    .format(' + '.join(selected)))
    kw2 = {'bootstrap': False, 'max_features': 0.333,
           'min_impurity_decrease': 1e-10, 'n_estimators': 20}

    # Stage 3
    rfr_formula3 = ('net_loss_pct ~ -1 + {0}'
                    .format(' + '.join(selected)))
    kw3 = {'n_estimators': 50, 'max_features': 0.1,
           'min_impurity_decrease': 1e-10}

    # define model_choice
    model_specs = [['GBC', gbm_formula1, kw1],
                   ['RFR', rfr_formula2, kw2],
                   ['RFR', rfr_formula3, kw3]]

    # create new folder if not exist
    if not os.path.exists(EXPORT_PATH):
        print('\nCreating results directory...')
        os.makedirs(EXPORT_PATH)
    output_dir_name = '.'.join([m[0] for m in model_specs] +
                               [FILENAME] +
                               [datetime.datetime.now()
                                .strftime("%y%m%d%H%M%S")])
    output_dir_path = os.path.join(EXPORT_PATH, output_dir_name)
    if not os.path.exists(output_dir_path):
        print('\nCreating directory at export location...')
        os.makedirs(output_dir_path)

    for train, test, trial_i in utility.train_test_splitter(df_low,
                                                            0.1, 'ORIG_DTE'):
        train_pos = train[train['dflt_pct'] > 0]

        # stage 1
        train['PD_pred'], test['PD_pred'], model1 = fit_stage1(model_specs[0],
                                                               train, test, True,
                                                               return_model=True)
        # stage 2
        train['EAD_pred'], test['EAD_pred'], model2 = fit_stage2(model_specs[1],
                                                                 train_pos, test,
                                                                 True, train,
                                                                 return_model=True)
        # stage 3
        train['LGD_pred'], test['LGD_pred'], model3 = fit_stage3(model_specs[2],
                                                                 train_pos, test,
                                                                 True, train,
                                                                 return_model=True)

        # plot model summaries (just the first 10 most important)
        print('\nGenerating summary plot...')
        summ1 = model1.summary()[:10]
        summ2 = model2.summary()[:10]
        summ3 = model3.summary()[:10]
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        ax = axes[0, 0]
        ax.bar(summ1['Name'], summ1['Importance'])
        ax.set_xticklabels(summ1['Name'], rotation=90)
        ax.set_ylabel('Feature Importance')        
        ax.set_title('GBC model for Probability of Default')

        ax = axes[0, 1]
        ax.bar(summ2['Name'], summ2['Importance'],
               yerr=summ2['Std'])
        ax.set_xticklabels(summ3['Name'], rotation=90)
        ax.set_ylabel('Feature Importance')        
        ax.set_title('RFR model for Exposure at Default')

        ax = axes[1, 0]
        ax.bar(summ3['Name'], summ3['Importance'],
               yerr=summ3['Std'])
        ax.set_xticklabels(summ3['Name'], rotation=90)
        ax.set_ylabel('Feature Importance')        
        ax.set_title('RFR model for Loss given Default')

        ax = axes[1, 1]
        ax.set_axis_off()
        # make a dataframe of parameters for each model
        param_list = [pd.DataFrame(
            {'Model': '{0}{1}'
             .format(model.__class__.__name__, i+1),
             'Param':
             list(model.params().keys()),
             'Value':
             list(model.params().values()),
            })
            for i, model in
                      enumerate([model1, model2, model3])]

        num_rows = 5
        param_df = pd.concat([df[:num_rows] for df in param_list])
        colors = [['#F4D03F'] * 3] * num_rows +\
            [['#F1948A'] * 3] * num_rows +\
            [['#7DCEA0'] * 3] * num_rows
        tab = ax.table(cellText=param_df.values,
                       bbox=[0,0,1,1.05],
                       cellColours=colors,
                       colLabels=param_df.columns,
                       colWidths=[0.05, 0.1, 0.1])
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)

        fig.tight_layout()
        plotname = '{0}.model summary.png'.format(trial_i)
        fig.savefig(os.path.join(output_dir_path,
                                 plotname))                    
        plt.close('all')

        # bootstrapping
        train['L_pred'] = train['PD_pred'] * \
            train['EAD_pred'] * train['LGD_pred']
        test['L_pred'] = test['PD_pred'] * test['EAD_pred'] * test['LGD_pred']
        test_pos = test[test['dflt_pct'] > 0]

        # get bootstraps
        print('\nRunning bootstraps...')
        btstrp_size = 30
        btstrp_3stage = bootstrap.get_btstrp(fit_3stages, model_specs,
                                             train, test, btstrp_size)
        # stage 1 bootstraps
        btstrp_stage1 = bootstrap.get_btstrp(fit_stage1, model_specs[0],
                                             train, test, btstrp_size)
        # stage 2 bootstraps
        btstrp_stage2 = bootstrap.get_btstrp(fit_stage2, model_specs[1],
                                             train_pos, test_pos, btstrp_size)
        # stage 2 bootstraps
        btstrp_stage3 = bootstrap.get_btstrp(fit_stage3, model_specs[2],
                                             train_pos, test_pos, btstrp_size)

        vin_id, x, lo_name, hi_name = 'ORIG_DTE', 'AGE', '2.5%', '97.5%'

        # compile bootstrapped data for each stages
        df_all = pd.concat([test[[vin_id, x, 'final_loss_pct', 'L_pred']],
                            btstrp_3stage], axis=1)
        df_stage1 = pd.concat([test[[vin_id, x, 'min_dflt', 'PD_pred']],
                               btstrp_stage1], axis=1)
        df_stage2 = pd.concat([test_pos[[vin_id, x, 'dflt_pct', 'EAD_pred']],
                               btstrp_stage2], axis=1)
        df_stage3 = pd.concat([test_pos[[vin_id, x,
                                         'net_loss_pct', 'LGD_pred']],
                               btstrp_stage3], axis=1)

        
        for date, dall in df_all.groupby(vin_id):
            print('\nGenerating plots for {0}...'.format(date))
            ds = []
            for df_ in [df_stage1, df_stage2, df_stage3]:
                to_append = df_[df_[vin_id] == date]
                # remove vin_id column
                del to_append[vin_id]
                ds.append(to_append)
            del dall[vin_id]

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # stage 1 roc
            ax, d1 = axes[0, 0], ds[0]
            fpr, tpr = utility.get_roc_curve(train['min_dflt'], train['PD_pred'])
            roc_auc = utility.get_auc(fpr, tpr)
            ax.plot(fpr, tpr, label='Train AUC = {0:.2f}'.format(roc_auc),
                    **PLOT_PARAMS['train roc curve'])

            fpr, tpr = utility.get_roc_curve(d1['min_dflt'], d1['PD_pred'])
            roc_auc = utility.get_auc(fpr, tpr)
            ax.plot(fpr, tpr, label='Test AUC = {0:.2f}'.format(roc_auc),
                    **PLOT_PARAMS['test roc curve'])

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
            ax.scatter(c, d,
                       label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
                       **PLOT_PARAMS['train'])
            a, b = d2['EAD_pred'], d2['dflt_pct']
            ax.scatter(a, b,
                       label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
                       **PLOT_PARAMS['test'])
            xymax = min(0.5, max(a.max(), b.max()) * 1.05)
            xymin = max(-0.01, min(a.min(), b.min()) * 0.95)
            ax.set_xlim([xymin, xymax])
            ax.set_ylim([xymin, xymax])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Exposure at Default: % balance at default')
            ax.legend(loc="lower right")

            # # stage 2 bootstrap actual vs. predicted
            # ax, d2 = axes[0, 1], ds[1]
            # bootstrap.plot_btstrp(d2, 'dflt_pct', 'dflt_pct', 'EAD_pred',
            #                       lo_name, hi_name, ax=ax)
            # ax.set_xlabel('Actual')
            # ax.set_ylabel('Predicted')
            # ax.set_title('Exposure at Default: % balance at default')
            # ax.legend(loc="lower right")

            # stage 3 predicted vs. actual
            ax, d3 = axes[1, 0], ds[2]

            ax.plot([-10, 10], [-10, 10], **PLOT_PARAMS['actual line'])
            c, d = train['LGD_pred'], train['net_loss_pct']
            ax.scatter(c, d,
                       label='Train RMSE: {0:.2E}'.format(((c-d)**2).mean()),
                       **PLOT_PARAMS['train'])
            a, b = d3['LGD_pred'], d3['net_loss_pct']
            ax.scatter(a, b,
                       label='Test RMSE: {0:.2E}'.format(((a-b)**2).mean()),
                       **PLOT_PARAMS['test'])

            xymax = min(2, max(a.max(), b.max()) * 1.05)
            xymin = max(-0.5, min(a.min(), b.min()) * 0.95)
            ax.set_xlim([xymin, xymax])
            ax.set_ylim([xymin, xymax])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Loss given Default: % lost at default')
            ax.legend(loc="lower right")

            # # stage 3 bootstrap actual vs. predicted
            # ax, d3 = axes[1, 0], ds[2]
            # bootstrap.plot_btstrp(d3, 'net_loss_pct', 'net_loss_pct', 'LGD_pred',
            #                       lo_name, hi_name, ax=ax)
            # ax.set_xlabel('Actual')
            # ax.set_ylabel('Predicted')
            # ax.set_title('Loss given Default: % lost at default')
            # ax.legend(loc="lower right")

            # 3 stage bootstrap cumulative
            ax = axes[1, 1]
            cumsum = dall.set_index(x).cumsum()
            bootstrap.plot_btstrp(cumsum.reset_index(), x, 'final_loss_pct',
                                  'L_pred', lo_name, hi_name, ax=ax)
            ax.set_xlabel('Age of loan')
            ax.set_ylabel('% of ORIG_AMT lost')
            loan_count = test.loc[test.ORIG_DTE ==
                                  date, 'LOAN_ID_count'].values[0]
            ax.set_title(
                'Projected Loss: {0} loans from {1}'.format(loan_count, date))
            ax.legend(loc="lower right")
            # plot settings
            plt.tight_layout()
            plotname = '{2}.{1}.{0}.png'.format(loan_count, date, trial_i)
            plt.savefig(os.path.join(output_dir_path, plotname),
                        dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probability of default model for vintage data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataname', type=str, nargs='?', default='fannie_mae_data',
                        help='name of data folder')
    parser.add_argument('filename', type=str, help='name of data file')
    args = parser.parse_args(['vintage_filelist_50.csv'])
    main(args)

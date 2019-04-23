# Vintage Analysis

## Vintage Aggregation

Vintage level analysis requires loans to be aggregated by origination month. Loan information such as credit score will be weight-averaged by original amount of the loan. The aggregated data will be placed in the directory called &lsquo;vintage\_data&rsquo; within the data directory. To preprocess the data, run the following in command line:

    usage: vintage_aggregate.py [-h] [-l LOWER_LIM]
                                [dataname] [filelist] [varlist]
    
    Aggregate loan data by vintage.
    
    positional arguments:
      dataname              name of data folder (default: fannie_mae_data)
      filelist              text file containing list of file batches to read
                            (default: filelist.txt)
      varlist               text file containing a row of categorical and a row of
                            continuous loan variables (default: varlist.txt)
    
    optional arguments:
      -h, --help            show this help message and exit
      -l LOWER_LIM, --lower_lim LOWER_LIM
                            minimum number of loans per vintage (default: 500)

-   filelist: A textfile specifying file batches. The first element of each row must be the year for the batch. The remaining elements are the filenames in that batch. One batch will be read at a time. Place this file in the data directory.
-   varlist: A textfile specifying the optional loan variables. The first element of each row must be the keyword &lsquo;CAT&rsquo; for categorical variables or &lsquo;CONT&rsquo; for continuous variables. The remaining elements are the names of the columns.

A simple execution of aggregating the fannie mae dataset will look like the following:

    python vintage_aggregate.py -l LOWER_LIM

Running the code will create a new directory in the data directory called &lsquo;vintage\_analysis&rsquo;. Vintage aggregated data will be placed inside this directory under &lsquo;data&rsquo;.

    ├── data                    # Contains all data needed for analysis
    │   ├── data directory      # Name of custom dataset
    │   │   ├── raw             # Data before any preprocessing
    │   │   ├── clean           # Data after preprocessing
    │   │   ├── vintage_analysis
    │   │   │   ├── data        # Contains vintage-aggregated data
    │   │   │   └── results     # Results from running vintage modelling code
    └── ...

## Modeling

With the dataset in the format we want, run the following to get model results:
```
usage: vintage_analysis.py [-h] [-s DO_SAVE] [dataname] filename

Probability of default model for vintage data.

positional arguments:
  dataname              name of data folder (default: fannie_mae_data)
  filename              name of data file

optional arguments:
  -h, --help            show this help message and exit
  -s                    if set to True, will save trained model and test/training sets for future
                        use (default: False)
```

A simple execution will look like:
```
python vintage_analysis.py FILENAME -s True
```

-   Model specifications

We refer the user to vintage_notes.pdf for more details about the model. Essentially, we break up the prediction of net loss problem into 3 parts. The first stage is a binary classification problem predicting on which months each vintage has at least 1 defaulting loan. The second stage is a continuous problem predicting the % of ORIG_AMT that defaulted on those months. The last stage predicts the % of the defaulted amount that is lost as a result of resale of the loan. By default, the model uses a Gradient Boosting Classifier for the first stage and Random Forest Regression for stages 2 and 3. The default model specifications are:
```
Variables used:
['CSCORE_MN_wm', 'CSCORE_MN_wv', 'DTI_wm', 'DTI_wv', 'LOAN_ID_count',
 'OCLTV_wm', 'OCLTV_wv', 'ORIG_AMT_sum',
 'ORIG_CHN_C_wm', 'ORIG_CHN_R_wm', 'ORIG_RT_wm', 'ORIG_RT_wv',
 'PURPOSE_P_wm', 'PURPOSE_P_wv', 'PURPOSE_R_wm', 'PURPOSE_R_wv',
 'CPI', 'UNEMP', 'HPI', 'rGDP', 'AGE']
GBC: {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 0.5, 'min_impurity_decrease': 0.0001, 'n_estimators': 10, 'subsample': 0.1}
RFR: {'bootstrap': False, 'max_features': 0.333, 'min_impurity_decrease': 1e-10, 'n_estimators': 20}
```

-   Output

    All results such as plots that are generated will be placed in &lsquo;vintage\_analysis&rsquo; under a newly created directory called &lsquo;results&rsquo;.
    Saved model will be placed in &lsquo;vintage\_analysis/data&rsquo; under a newly created directory called &lsquo;saved_model&rsquo;

## Loading and Reusing Trained Models

We provide the option to save and load a trained model for making predictions on new datasets. Saving is implemented when you provide a flag to vintage_analysis.py. Loading is done by &lsquo;vintage\_load.py&rsquo; and implemented as follows:

```
usage: vintage_load.py [-h] [-d dataname] [-m modelname]

Load a trained model and make predictions on a new dataset, if provided.

positional arguments:
  -m modelname              name of saved model. Defaults to most recent save (default: None)
  -d dataname               name of new test data to make predictions on (default: None)

optional arguments:

  -h, --help            show this help message and exit

```
-   New data
    
    User should move the new data they want to test to &lsquo;vintage\_analysis/data&rsquo;
    
    All results generated by this new data will be placed in &lsquo;vintage\_analysis/results&rsquo;
    
## Compare the prediction under different macroeconomic trend

When user using the trained model to predict new datasets in &lsquo;eco\_load.py&rsquo;, the program will load every macroeconomic trend data under &lsquo;data/economic/eco\_df&rsquo;, the user can compare different prediction under different macroeconomic trend.

```
usage: eco_load.py [-h] [-d dataname] [-m modelname]

combine the new dataset with macroeconomic data, using trained model and make predictions on each macroeconomic trend.

positional arguments:
  -m modelname              name of saved model. Defaults to most recent save (default: None)
  -d dataname               name of new test data to make predictions on (default: None)

optional arguments:
  -h, --help            show this help message and exit

```
-   New data
    
    User should move the new data they want to test to &lsquo;vintage\_analysis/data&rsquo;

-   Macroeconomic trend

    User should move all macroeconomic trend they want to compare to &lsquo;data/economic/eco\_df&rsquo;
    
    All results generated by this new data will be placed in &lsquo;vintage\_analysis/results&rsquo;
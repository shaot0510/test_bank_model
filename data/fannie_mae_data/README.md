# Preprocessing Fannie Mae

First, download the fannie mae data from [the site](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) by clicking &lsquo;Access the Data&rsquo; on the right-hand side. A new account is required. We recommend downloading the &lsquo;Entire Single Family Eligible Fixed Rate Mortgage Dataset&rsquo;, which is about 24GB. You may initially download just a few quarters to ensure the code is functioning properly. Unzip all downloaded files. In each unzipped folder, there should be a pair of txt files, Acquisition and Performance, the names of which are in the following format:

    Acquisition_2001Q2.txt
    Performance_2001Q2.txt

Place all &lsquo;Acquistion&rsquo; and &lsquo;Performance&rsquo; files in &lsquo;raw&rsquo; in the data directory (&lsquo;fannie\_mae\_data&rsquo;), which you will need to create yourself:

    data/fannie_mae_data/raw

In order to preprocess the data, run the following line in the command line:

    Rscript --vanilla cleaning_initial_with_loss.R -n NUM_LOANS_TO_READ -c CHUNK_SIZE -f HAS_WC

- NUM_LOANS_TO_READ: only read this many unique loans will be read from each raw file. The process stops after the number of loans read surpasses this number.
- CHUNK_SIZE: This many rows will be read from the raw file at a time. For example, if set to 1 million, random points in the raw file will be chosen and 1 million subsequent rows will be read in.
- HAS_WC: If wc is available on your system, the code will use this to estimate the number of rows in the raw file without reading the entire file. If wc is not available, it uses readr, which will take longer.

## Output

The code automatically creates a new folder called &lsquo;clean&rsquo; in the data directory and places the preprocessed dataset in this folder. The name of the file will be 'NUM_LOANS_TO_READ.QUARTER.csv', '10000.2010Q1.csv' for example.
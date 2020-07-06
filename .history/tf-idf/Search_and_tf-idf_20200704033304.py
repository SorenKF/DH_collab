# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# 
# %% [markdown]
# Here is my extended search function fromte text processing assignment

# %%
import jupyternotify
ip = get_ipython()
ip.register_magics(jupyternotify.JupyterNotifyMagics)

get_ipython().run_line_magic('load_ext', 'jupyternotify')

# test
#%notify
#import time
#time.sleep(5)


# %%
from notify_run import Notify


# %%
notify = Notify()


# %%
notify.register()


# %%
import re, glob, os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import math


# %%
# ver. 3 - debugging the count of regexes.

### NOTE ###
# - The code counts terms twice, if they are similar within the same pass. This means that drunk* and drun* count 1 and 2, because the loop counts drunk* again on the second iteration. THIS IS A PROBLEM WITH DRUNK* AND DRUNKARD!!!
# -  

def regex_search(folder_path , regList):
    """
    Search a folder path for a given regex.

    :folder_path: a string with the path to a dir with processed .txt files 
    :regList: a list of regular expressions as strings.
    """
    resultsDict = defaultdict(lambda: defaultdict(dict)) #consider using regular dict for the inner layer, since you know which terms you are searching.


    # file_names = [] 

    # compile any search term(s) given for searching.
    #search_terms = re.compile('|'.join(regList)) # with word boundaries: re.compile(r'\b(?:%s)\b' % '|'.join(regList))
    search_terms_re = [re.compile(reg) for reg in regList]

        
    #loop through the folder with txt files.
    
    for filepath in tqdm(list(glob.iglob(folder_path + "*\\*", recursive=True))):  #consider using os.listdir() for a subset of files
        filename = os.path.basename(filepath)
        
        
        #open the file
        with open(filepath, "r") as infile:
            content = infile.read()

        #find the document length (total tokens)
        resultsDict[filename]['text_length'] = len(re.findall(r'\w+', content)) #consider using the token pattern from https://github.com/gearmonkey/tfidf-python/blob/master/tfidf.py : re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+")

        #search for the terms in the file, given the list of regexes.
        for term in search_terms_re:
            n_hits = 0
            for match in re.finditer(term, content):
                #print(term, match)
                n_hits += 1
                #print(n_hits)

        # add the count of hits to the nested results dictionary with readable versions of the terms. 
                resultsDict[filename][re.sub(r'\\.*$', '', str(term.pattern))] = n_hits 



    #export to df
    resultsDF = pd.DataFrame.from_dict(resultsDict, orient='index').fillna(0)
    # (re)name the index column
    resultsDF = resultsDF.rename_axis('file_id')
    # sum counts for the term columns
    resultsDF['terms_sum'] = resultsDF.iloc[:, 1:].sum(axis=1)  # https://stackoverflow.com/questions/48923460/how-do-i-sum-a-column-range-in-python
    # trim output to only include docs with hits from the search
    trimmedOutput = resultsDF[(resultsDF['terms_sum'] > 0)]

    # compute number of hits (in terms of docs) / total corpus size
    percentageHits = int((len(trimmedOutput) / len(resultsDF)) * 100)
    #print(f'percentage of files found in the total corpus on this query:  {percentageHits}%')
    
    # Add a bottom row with totals for all columns
    trimmedOutput.loc[f'Total ({len(trimmedOutput)} doc(s) = {percentageHits}% of corpus)'] = trimmedOutput.sum(numeric_only=True, axis=0)
    
    # return both the raw output and the trimemd output containing only the files with hits.
    return resultsDF, trimmedOutput

#print(output)
#print(list(results.items())[:10])


#### for this code I used solutions from:
# https://stackoverflow.com/questions/6750240/how-to-do-re-compile-with-a-list-in-python
# https://howchoo.com/g/yjjknjdinmq/nested-defaultdict-python
# https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index 
# https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/


# %%
searchQuery = [
                "drunk\w*",
                "intoxicat\-?\w*",
                "temperance",
                "beer\-?\w*",
                "liquor",
                "alcohol\-?\w*",
                "brandy",
                "booz\-?\w*",
                "drunkard",
                "by\-drink\w*"]


# %%

#function call

rawOutput, trimmedOutput = regex_search("C://TEMPORARY/fullOB/" , searchQuery)
get_ipython().run_line_magic('notify', '')


# %%
# inspect the trimmed output
trimmedOutput


# %%
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('multiple_results.xlsx', engine='xlsxwriter')


# %%
# write the TRIMMED  output (only containing search hits) to a human friendly files

trimmedOutput.to_csv('trimmed_results.tsv', sep="\t") # index_label='file_id' 
trimmedOutput.to_excel(writer, sheet_name='trimmed_results')
rawOutput.to_excel(writer, sheet_name='raw_results')
writer.save()

# %% [markdown]
# ## Calculating TF-IDF for the search results
# Next, we load the data, and calculate the tf-df for each term per doc, and find their averages per doc.

# %%
# read the output of the last run, drop the index column and the footer (containing sum totals).
# consider reading multiple excel sheets.
xls_in = pd.ExcelFile('multiple_results.xlsx')
df_in = pd.read_excel(xls_in, 'trimmed_results', index_col=False, skipfooter=1) #read the trimmed results without the footer
df_in.set_index('file_id' , inplace=True) # drop the added index.
del df_in['terms_sum'] # remove the totals for all terms taken together.


# %%
# inspect the loaded data

df_in


# %%
df_tfidf = pd.DataFrame # index= the header in df_in , colum should be tf , idf , tf-idf, average (tfidf) <- this is basically just flipping part of the other dataframe around on all rows, colum 1:-1. transpose? 

# tf = df_in[['text_length' , 'drunk*']].sum(axis=1)
normalized_tf = df_in.iloc[:, 1:].divide(df_in['text_length'] , axis=0)


#idf = len(df_in).divide(df_in.iloc[:, 1:] , axis=0) ## put this in the new df for each row

#idf

normalized_tf = normalized_tf.transpose()
normalized_tf
# consider writing a function for each variable here and using .apply to insert the resulting computations in new columns.
# otherwise, load it back into another container (dict of dicts and compute things from there before moving back to dataframe)

#idf = len(df_in['file_id'] / column_values >=1) ### total documents / total document with term t in them.
#df_in['tf_idf'] = tf*idf


# %%
def get_tf(term_column):
    """
    gets
    """


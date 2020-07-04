# Search_and_tf-idf
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Here is my reworked search search function.

# %%
import re, glob, os, winsound
from collections import defaultdict
import pandas as pd

import winsound

frequency = 440  # Set Frequency To 2500 Hertz
duration = 2000  # Set Duration To 1000 ms == 1 second


# %%
# ver. 2 - uses a list of regex terms.

### NOTE ###
# - The code counts terms twice, if they are similar within the same pass. This means that drunk* and drun* count 1 and 2, because the loop counts drunk* again on the second iteration. THIS IS A PROBLEM WITH DRUNK* AND DRUNKARD!!! check https://stackoverflow.com/questions/1374457/find-out-how-many-times-a-regex-matches-in-a-string-in-python
# -  

def regex_search(folder_path , regList, verbose=1):
    """
    Search a folder path for a given regex.

    :folder_path: a string with the path to a dir with processed .txt files 
    :regList: a list of regular expressions as strings.
    """
    resultsDict = defaultdict(lambda: defaultdict(dict)) #consider using regular dict for the inner layer, since you know which terms you are searching.


    file_names = []

    # compile any search term(s) given for searching.
    #search_terms = re.compile('|'.join(regList)) # with word boundaries: re.compile(r'\b(?:%s)\b' % '|'.join(regList))
    search_terms_re = [re.compile(reg) for reg in regList]


    #loop through the folder with txt files.
    for filepath in glob.iglob(folder_path + "/*"):  #for future reference, if you just want to iterate through a number of files, use os.listdir() and slice it [:20] for instance.
        n_hits = 0
        filename = os.path.basename(filepath)
        #file_names.append(filename)
        
        #open the file
        with open(filepath, "r") as infile:
            content = infile.read()

        #find the document length (total tokens)
        resultsDict[filename]['text_length'] = len(re.findall(r'\w+', content))

        #search for the terms in the file, given the list of regexes.
        for term in search_terms_re:
            if term.findall(content):
                n_hits += 1
           

        # add the coutn of hits to the nested results dictionary 
            resultsDict[filename][term.pattern] = n_hits


    #export to df
    resultsDF = pd.DataFrame.from_dict(resultsDict, orient='index')
    # (re)name the index column
    #resultsDF = resultsDF.rename_axis('file_id')

    return resultsDF

#print(output)
#print(list(results.items())[:10])


#### for this code I used solutions from:
# https://stackoverflow.com/questions/6750240/how-to-do-re-compile-with-a-list-in-python
# https://howchoo.com/g/yjjknjdinmq/nested-defaultdict-python
# https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index 
# https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/


# %%
searchQuery = [
                "drunk*",
                "intoxicat*",
                "temperance",
                "beer*",
                "liquor",
                "alcohol*",
                "brandy",
                "booz*",
                "drunkard",
                "by-drink*"]


# %%

#function call

output = regex_search("C:/TEMPORARY/18" , searchQuery)


# %%
# inspect the dataframe
output


# %%
# write the output to a human friendly files

output.to_csv('results.tsv', sep="\t", index_label='file_id')
output.to_excel('results.xlsx', index_label='file_id')

winsound.Beep(frequency, duration)
#winsound.MessageBeep()

# %% [markdown]
# Get the counts for tf-idf

# %%



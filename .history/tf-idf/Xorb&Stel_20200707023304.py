import re, glob, os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

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
        with open(filepath, "r", encoding = 'utf-8') as infile:
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


#function call

rawOutput, trimmedOutput = regex_search("C:/Users/Stell/Downloads/OB_FULL" , searchQuery)

# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('multiple_results.xlsx', engine='xlsxwriter')

# # write the TRIMMED  output (only containing search hits) to a human friendly files
# trimmedOutput.to_csv('results_inspect.tsv', sep="\t") # index_label='file_id' 
# trimmedOutput.to_excel(writer, sheet_name='blabla')
# rawOutput.to_excel(writer, sheet_name='raw_blabla')
# writer.save()


#trimmedOutput.to_csv('freq_dict_inspect2.tsv', sep="\t", index_label='file_id')
trimmedOutput.to_excel('freq_dict.xlsx', index_label='file_id')

df_results = pd.read_excel("freq_dict.xlsx")

def get_tf_idf(final_df, freq_df, term):
    """
    Blablabla
    """
    
    final_df[term+'_freq'] = freq_df[term]
    final_df['tf_'+term] = (final_df[str(term+'_freq')] / final_df['text_length'])
    
    counter = 0
    for index, freq in enumerate(final_df[term+'_freq']):
        if final_df.iloc[index][term+'_freq'] != 0:
            counter+= 1
        
    final_df['idf_'+term] = (math.log((len(final_df['file_id']))/ counter))
    
    final_df['tf.idf_'+term] = final_df['tf_'+term] * final_df['idf_'+term]
    
    columns = ['tf_'+term, 'idf_'+term, term+'_freq']
    final_df = final_df.drop(columns, axis=1)
    return(final_df)

final_df = pd.DataFrame()
final_df['file_id'] = df_results['file_id']
final_df['text_length'] = df_results['text_length']

freq_df = df_results

final_df = get_tf_idf(final_df, freq_df, 'drunk')
final_df = get_tf_idf(final_df, freq_df, 'beer')
final_df = get_tf_idf(final_df, freq_df, 'intoxicat')
final_df = get_tf_idf(final_df, freq_df, 'temperance')
final_df = get_tf_idf(final_df, freq_df, 'liquor')
final_df = get_tf_idf(final_df, freq_df, 'alcohol')
final_df = get_tf_idf(final_df, freq_df, 'brandy')
final_df = get_tf_idf(final_df, freq_df, 'booz')
final_df = get_tf_idf(final_df, freq_df, 'drunkard')
#final_df = get_tf_idf(final_df, freq_df, 'by-drink')

# two ways of getting either the arithmetical average:
average = (final_df.iloc[: ,3 :].sum(axis=1) / len((final_df.iloc[: ,3 :].columns)))

final_df.to_excel('tf_idf_Soren2.xlsx', index_label='file_id')
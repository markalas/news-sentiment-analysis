import os
import pandas as pd
import numpy as np
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Setup working directory and relative filepaths
current_dir = os.curdir
data_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "data_clean")

# Get data path
abc_data_dir = os.path.join(data_dir, "abc-news.csv")
bbc_data_dir = os.path.join(data_dir, "bbc-news.csv")
cbs_data_dir = os.path.join(data_dir, "cbs-news.csv")
cnn_data_dir = os.path.join(data_dir, "cnn-news.csv")

# Read csv data
abc_data_csv = pd.read_csv(abc_data_dir, encoding="utf-16")
bbc_data_csv = pd.read_csv(bbc_data_dir, encoding="utf-16")
cbs_data_csv = pd.read_csv(cbs_data_dir, encoding="utf-16")
cnn_data_csv = pd.read_csv(cnn_data_dir, encoding="utf-16")

# Make csv to DataFrame
abc_data_df = pd.DataFrame(abc_data_csv.copy())
bbc_data_df = pd.DataFrame(bbc_data_csv.copy())
cbs_data_df = pd.DataFrame(cbs_data_csv.copy())
cnn_data_df = pd.DataFrame(cnn_data_csv.copy())

# Remove columns from Dataframe
col_ls = ['id', 'page_id', 'message', 'description', 'caption', 'link', 'picture']
abc_data_df.drop(columns = col_ls, inplace = True)
bbc_data_df.drop(columns = col_ls, inplace = True)
cbs_data_df.drop(columns = col_ls, inplace = True)
cnn_data_df.drop(columns = col_ls, inplace = True)

# Determine how many post_types and status_types and factorize types
post_types_unique = pd.unique(abc_data_df['post_type'])
post_types_factorize = pd.factorize(post_types_unique, sort=True)[0]
status_types_unique = pd.unique(abc_data_df['status_type'])
status_types_unique = [i for i in status_types_unique if str(i) != 'nan']
print(status_types_unique)
status_types_factorize = pd.factorize(status_types_unique, sort=True)[0]

# Create dictionary for post_types and status_types
post_types_dict = dict(zip(post_types_unique, post_types_factorize))
status_types_dict = dict(zip(status_types_unique, status_types_factorize))

# Define function to assign kv pairs to post_types and status_types
def post_cat(x):
    for type, category in post_types_dict.items():
        if x in type:
            return category

def status_cat(x):
    for type, category in status_types_dict.items():
        try:
            if x in type:
                return category
        except TypeError:
            return -1

# Apply function to DataFrame
abc_data_df['post_category'] = abc_data_df['post_type'].apply(post_cat)
abc_data_df['status_category'] = abc_data_df['status_type'].apply(status_cat)

# Calculate total reacts
react_col = ['likes_count', 'comments_count', 'shares_count', 'love_count',
       'wow_count', 'haha_count', 'sad_count', 'thankful_count', 'angry_count',]
       
abc_data_df['total_reacts'] = abc_data_df[react_col].sum(axis=1)

# ToDo: Create Class to clean all dataframes at once


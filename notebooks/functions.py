import pandas as pd 
from collections import Counter
from itertools import chain
import re

## NOTEBOOK: K-MEANS CLUSTERING

def pack_items(user_or_order_id, df, df_col: str):
    """
    Function to pack products, aisles or departments by unique user or order.
    """
    lst = []
    for i in user_or_order_id:
        x = df.loc[df['user_id'] == i][df_col].unique()
        lst.append(x)
    return lst


def top30_products(num_cluster, df):
    """
    Function to get the first 30 products most purchased for all
    customers that belong to one cluster (num_cluster).
    """
    prods_lst = [ele for ele in df[df['cluster'] == num_cluster]['products']]
    prods_lst = list(chain.from_iterable(prods_lst))
    freq_dict = dict(Counter(prods_lst))
    sorted_prods = sorted(freq_dict.items(), key=lambda item:item[1], reverse=True)

    top30_prods = []
    for i in range(0,30):
        top30_prods.append(sorted_prods[0:30][i][0])
        
    return top30_prods


def top30_aisles(num_cluster, df):
    """
    Function to get the first 30 aisles most "visited" for all
    customers that belong to one cluster (num_cluster).
    """
    aisles_lst = [ele for ele in df[df['cluster'] == num_cluster]['aisles']]
    aisles_lst = list(chain.from_iterable(aisles_lst))
    freq_dict = dict(Counter(aisles_lst))
    sorted_prods = sorted(freq_dict.items(), key=lambda item:item[1], reverse=True)

    top30_aisles = []
    for i in range(0,30):
        top30_aisles.append(sorted_prods[0:30][i][0])
        
    return top30_aisles




## NOTEBOOK: IMPROVED BASKET ANALYSIS

def pack_items_by_order(id_lst, df, df_col: str):
    """
    Function to pack products, aisles or departments by unique user or order.
    """
    lst = []
    for i in id_lst:
        x = df.loc[df['order_id'] == i][df_col].tolist()
        lst.append(x)
    return lst

def keywords_match(df_col, keywords_lst):
    lst = []
    pattern = "({})".format("|".join(keywords_lst))
    for i in df_col:
        x = re.findall(pattern, i)
        if len(x) == 0:
            lst.append('no match')
        else:
            lst.append(x)
    return lst


def get_items(row):
    if len(row) > 1:
        return row[-1]
    if len(row) == 1:
        return row[0]
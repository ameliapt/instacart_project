import pandas as pd 
from collections import Counter
from itertools import chain

## NOTEBOOK: K-MEANS CLUSTERING

def pack_items(user_id_lst, df, df_col: str):
    """
    Function to pack products, aisles or departments by unique user.
    """
    lst = []
    for user in user_id_lst:
        x = df.loc[df['user_id'] == user][df_col].unique()
        lst.append(x)
    return lst


def top30_products(num_cluster):
    """
    Function to get the first 30 products most purchased for all
    customers that belong to one cluster (num_cluster).
    """
    prods_lst = [ele for ele in inst_user[inst_user['cluster'] == num_cluster]['products']]
    prods_lst = list(chain.from_iterable(prods_lst))
    freq_dict = dict(Counter(prods_lst))
    sorted_prods = sorted(freq_dict.items(), key=lambda item:item[1], reverse=True)

    top30_prods = []
    for i in range(0,30):
        top30_prods.append(sorted_prods[0:30][i][0])
        
    return top30_prods


def top30_aisles(num_cluster):
    """
    Function to get the first 30 aisles most "visited" for all
    customers that belong to one cluster (num_cluster).
    """
    aisles_lst = [ele for ele in inst_user[inst_user['cluster'] == num_cluster]['aisles']]
    aisles_lst = list(chain.from_iterable(aisles_lst))
    freq_dict = dict(Counter(aisles_lst))
    sorted_prods = sorted(freq_dict.items(), key=lambda item:item[1], reverse=True)

    top30_aisles = []
    for i in range(0,30):
        top30_aisles.append(sorted_prods[0:30][i][0])
        
    return top30_aisles
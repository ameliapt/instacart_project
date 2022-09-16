import pandas as pd 
from collections import Counter
from itertools import chain
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pickle



## K-MEANS: MAIN FUNCTION

def label_data(file_path, path_to_store):

    print('Loading data...')
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df_km = df.copy()

    print('Preprocessing data...')
    df_km.drop(['reordered', 'product_id', 'department_id', 'aisle_id'], axis=1, inplace=True)
    df_1 = df_km[['user_id', 'days_since_prior_order']]
    df_2 = df_km[['department']]
    df_3 = df_km[['user_id','order_id','add_to_cart_order']]

    # Grouping data
    df_1 = df_km.groupby('user_id').agg({'days_since_prior_order':'mean'})

    # Encoding data
    filename = config['encoders']['ohe']
    with open(filename, "rb") as file:
            ohe = pickle.load(file)

    df_2_enc = ohe.transform(df_2).toarray()
    df_2 = pd.DataFrame(df_2_enc, columns = ohe.get_feature_names_out())
    df_2['user_id'] = df_km['user_id']
    df_2 = df_2.groupby('user_id').sum()
    df_2 = df_2[['department_produce', 'department_dairy eggs', 'department_frozen','department_snacks']]

    # Compute total number of purchases by user
    df_3 = df_3.groupby(['user_id', 'order_id']).max().reset_index()
    df_3 = df_3.groupby('user_id').agg({'add_to_cart_order':'sum'})

    # Combine columns into one dataframe
    X = df_1.join(df_2)
    X = X.join(df_3)

    # Handle NaNs
    X['days_since_prior_order'] = X['days_since_prior_order'].fillna(0)

    print('Scaling data...')
    filename = config['scalers']['minmax']
    with open(filename, "rb") as file:
            scaler = pickle.load(file)

    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)


    print('Applying K-Means algorithm with 7 clusters...')
    with open(config['models']['kmeans_path_k'], 'rb') as f:
        km7 = pickle.load(f)

    y_means = km7.labels_

    print('Gathering all information and assigning labels...')
    user_id = df['user_id'].unique()
    df_label = pd.DataFrame(user_id, columns = ['user_id'])
    df_label['cluster'] = y_means

    df = df.merge(df_label, how='left', on='user_id')

    print('Saving csv file with labels...')
    df.to_csv(path_to_store_csv)
    print('Done!')


## NOTEBOOK: BASKET ANALYSIS

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
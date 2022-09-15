import pandas as pd
import numpy as np
import pickle
import yaml

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

from functions import top30_products
from functions import top30_aisles
from functions import pack_items

try: 
    with open ("./../params.yaml", 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

file_path = config['data']['instacart_sample']
path_to_store_csv = config['data']['instacart_labels']

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


    print('Applying KMeans algorithm with 7 clusters...')
    with open(config['models']['kmeans_path_k'], 'rb') as f:
        km7 = pickle.load(f)

    y_means = km7.labels_

    print('Gathering all information for each user and assigning label...')
    user_id = df['user_id'].unique()
    prod_lst = pack_items(user_id, df, 'product_name')
    aisles_lst = pack_items(user_id, df, 'aisle')
    dept_lst = pack_items(user_id, df, 'department')

    df_user = df.groupby('user_id').agg({'order_number':'max', 'order_dow': pd.Series.mode,
                                     'order_hour_of_day': pd.Series.mode, 'days_since_prior_order': 'mean'}).round().reset_index()

    df_user['departments'] = dept_lst
    df_user['aisles'] = aisles_lst
    df_user['products'] = prod_lst
    df_user['num_products_purchased'] = df_3['add_to_cart_order'].tolist()
    df_user['cluster'] = y_means

    print('Saving csv file with labels...')
    df_user.to_csv(path_to_store_csv)
    print('Done!')


clustering = label_data(file_path, path_to_store_csv)
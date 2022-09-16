import pandas as pd 
from collections import Counter
from itertools import chain
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle


## K-MEANS: MAIN FUNCTION

def label_data(file_path, path_to_store):

    print('Loading data...')
    df = pd.read_csv(file_path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df_km = df.copy()

    print('Preprocessing data...')
    df_dept = df_km[['department']]

    # Encoding data
    filename = config['encoders']['ohe']
    with open(filename, "rb") as file:
            ohe = pickle.load(file)

    df_2_enc = ohe.transform(df_dept).toarray()
    df_2 = pd.DataFrame(df_2_enc, columns = ohe.get_feature_names_out())
    df_2['user_id'] = df_km['user_id']
    df_2 = df_2.groupby('user_id').sum()

    print('Scaling data...')
    filename = config['scalers']['minmax']
    with open(filename, "rb") as file:
            scaler = pickle.load(file)

    X_scaled = scaler.transform(df_2)
    X_scaled = pd.DataFrame(X_scaled, columns = df_2.columns)

    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    pca_samples_x = pca.transform(X_scaled)
    ps_x = pd.DataFrame(pca_samples_x, columns = ['PC1', 'PC2'])
    ps_x.head()

    kmeans = KMeans(n_clusters = 4,
                    random_state = config['models']['kmeans_randomstate'])
    kmeans.fit(ps_x)
    y_means = kmeans.labels_

    print('Gathering all information and assigning labels...')
    user_id = df_km['user_id'].unique()
    df_label = pd.DataFrame(user_id, columns = ['user_id'])
    df_label['cluster'] = y_means

    df_km = df_km.merge(df_label, how='left', on='user_id')

    print('Saving csv file with labels...')
    df_km.to_csv(path_to_store_csv)
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
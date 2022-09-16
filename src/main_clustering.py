import pandas as pd
import numpy as np
import pickle
import yaml
from functions import label_data

try: 
    with open ("./../params.yaml", 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

file_path = input('Introduce the path of the csv file:')
path_to_store = input('Introduce the path where you want to store the final df:')
clustering = label_data(file_path, path_to_store)

import os
import numpy
import pandas as pd
from urllib import request
import shutil
import zipfile
import kaggle
import gzip

# TDCE: Add OpenML support for LAW dataset
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("Warning: openml not installed. LAW dataset cannot be downloaded. Install with: pip install openml")


DATA_DIR = 'data'


NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'lending-club': 'wordsforthewise/lending-club',
    'gmc': 'GiveMeSomeCredit',
    #'dutch': 'https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/dutch.csv',
    'bank': 'https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/bank-full.csv',
}

# TDCE: OpenML dataset IDs
OPENML_DATASETS = {
    'law': 43890,  # Law School Admission Binary dataset
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):

    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

def download_from_repo(name):

    print(f'Start processing dataset {name} from repo.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        url = NAME_URL_DICT_UCI[name]
        # request.urlretrieve(url, f'{save_dir}/{name}.csv')
        df = pd.read_csv(url)
        df.to_csv(f'{save_dir}/{name}.csv')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
    else:
        print('Aready downloaded.')


def download_kaggle(name):
    print(f'Start processing dataset {name} from Kaggle.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        kaggle.api.authenticate()

        if name == 'gmc':
            kaggle.api.competition_download_files(url, path=save_dir)
            print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
            unzip_file(f'{save_dir}/{url}.zip', save_dir)
        else:
            kaggle.api.dataset_download_files(url, path=save_dir, unzip=False)
            print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
            unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Already downloaded.')


def download_from_openml(name):
    """Download dataset from OpenML
    
    Args:
        name: Dataset name (must be in OPENML_DATASETS)
    """
    if not OPENML_AVAILABLE:
        raise ImportError("openml package not installed. Install with: pip install openml")
    
    if name not in OPENML_DATASETS:
        raise ValueError(f"Dataset {name} not found in OPENML_DATASETS")
    
    print(f'Start downloading dataset {name} from OpenML.')
    save_dir = f'{DATA_DIR}/{name}'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        dataset_id = OPENML_DATASETS[name]
        dataset = openml.datasets.get_dataset(dataset_id)
        
        # Get data
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Combine features and target
        df = pd.DataFrame(X)
        df[dataset.default_target_attribute] = y
        
        # Save to CSV
        csv_path = os.path.join(save_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)
        
        print(f'Finish downloading dataset {name} (OpenML ID: {dataset_id}), data has been saved to {save_dir}.')
    else:
        print('Already downloaded.')


if __name__ == '__main__':
    for name in NAME_URL_DICT_UCI.keys():
        if name in ['lending-club', 'gmc']:
            download_kaggle(name)
        elif name in ['bank', 'dutch']:
            download_from_repo(name)
        else:
            download_from_uci(name)
    
    # TDCE: Download LAW dataset from OpenML
    if 'law' in OPENML_DATASETS:
        try:
            download_from_openml('law')
        except Exception as e:
            print(f"Failed to download LAW dataset: {e}")
            print("You can manually download from: https://www.openml.org/d/43890")
    
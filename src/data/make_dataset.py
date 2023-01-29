# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import os
import git
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")

def read_ranking(file,
                 input_data_dir=os.path.join(git_root,'data','external')):

    with open(os.path.join(input_data_dir,file), 'r') as f:
        ranking = [line.strip() for line in f.readlines()]

    return ranking

def read_files():

    files = [f for f in os.listdir(os.path.join(git_root,'data','external')) if f.endswith('.txt')]

    raw_data = {os.path.splitext(file)[0]: {'ranking': read_ranking(file)} for file in files}

    return raw_data

def raw_data_to_df(raw_data):

    people = list(raw_data.keys())
    people_le = LabelEncoder()
    people_le.fit(people)
    
    chocs = raw_data[next(iter(raw_data))]['ranking']
    choc_le = LabelEncoder()
    choc_le.fit(chocs)

    ranking_df = pd.DataFrame.from_dict(raw_data).melt(var_name='person',value_name='choc').explode('choc')

    ranking_df['person_idx'] = people_le.transform(ranking_df['person'])
    ranking_df['choc_idx'] = choc_le.transform(ranking_df['choc'])

    ranking_df['rank'] = ranking_df.groupby('person').cumcount()

    return ranking_df

def read_processed_data():

    with open(os.path.join(git_root, 'data','processed','ranking_df.pkl'), 'rb') as f:
        ranking_df = pickle.load(f)

    return ranking_df

import os
import git
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")

def read_ranking(file,
                 input_data_dir=os.path.join(git_root,'data','external')):

    with open(os.path.join(input_data_dir,file), 'r') as f:
        ranking = [line.strip() for line in f.readlines()]

    return ranking

def read_files():

    files = [f for f in os.listdir(os.path.join(git_root,'data','external')) if f.endswith('.txt')]

    raw_data = {os.path.splitext(file)[0]: {'ranking': read_ranking(file)} for file in files}

    return raw_data

def raw_data_to_df(raw_data):

    people = list(raw_data.keys())
    people_le = LabelEncoder()
    people_le.fit(people)
    
    chocs = raw_data[next(iter(raw_data))]['ranking']
    choc_le = LabelEncoder()
    choc_le.fit(chocs)

    ranking_df = pd.DataFrame.from_dict(raw_data).melt(var_name='person',value_name='choc').explode('choc')

    ranking_df['person_idx'] = people_le.transform(ranking_df['person'])
    ranking_df['choc_idx'] = choc_le.transform(ranking_df['choc'])

    ranking_df['rank'] = ranking_df.groupby('person').cumcount()

    return ranking_df

def read_processed_data():

    with open(os.path.join(git_root, 'data','processed','ranking_df.pkl'), 'rb') as f:
        ranking_df = pickle.load(f)

    return ranking_df

import os
import git
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


git_root = git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")

def read_ranking(file,
                 input_data_dir=os.path.join(git_root,'data','external')):

    with open(os.path.join(input_data_dir,file), 'r') as f:
        ranking = [line.strip() for line in f.readlines()]

    return ranking

def read_files():

    files = [f for f in os.listdir(os.path.join(git_root,'data','external')) if f.endswith('.txt')]

    raw_data = {os.path.splitext(file)[0]: {'ranking': read_ranking(file)} for file in files}

    return raw_data

def raw_data_to_df(raw_data):

    people = list(raw_data.keys())
    people_le = LabelEncoder()
    people_le.fit(people)
    
    chocs = raw_data[next(iter(raw_data))]['ranking']
    choc_le = LabelEncoder()
    choc_le.fit(chocs)

    ranking_df = pd.DataFrame.from_dict(raw_data).melt(var_name='person',value_name='choc').explode('choc')

    ranking_df['person_idx'] = people_le.transform(ranking_df['person'])
    ranking_df['choc_idx'] = choc_le.transform(ranking_df['choc'])

    ranking_df['rank'] = ranking_df.groupby('person').cumcount()

    return ranking_df

def read_processed_data():

    with open(os.path.join(git_root, 'data','processed','ranking_df.pkl'), 'rb') as f:
        ranking_df = pickle.load(f)

    return ranking_df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    raw_data = read_files()
    ranking_df = raw_data_to_df(raw_data)

    with open(os.path.join(git_root, 'data','processed','ranking_df.pkl'), 'wb') as f:
        pickle.dump(ranking_df, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

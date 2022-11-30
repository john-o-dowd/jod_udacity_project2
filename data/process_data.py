import sys
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
plt.ion()

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files and merge to a single pandas dataframe (after dropping unhelpful columns)
    :param messages_filepath: filepath to message csv file
    :param categories_filepath: filepath to categories csv file
    :return: df_merged: dataframe with merged categories and messages
    """
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df_merged = pd.merge(df_messages, df_categories, left_index=True, right_index=True)
    categories = df_merged.categories.str.split(';', expand=True)
    category_colnames = categories.loc[0].values
    category_colnames = list(map(lambda x: x[0:-2], category_colnames)) # use the string part of category values to get column names
    categories.columns = category_colnames  # add names to the category columns

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract(r'\-(.*)').astype('int')  # extract first number found in each category

    categories.set_index(df_merged['id_x'])
    df_merged.drop('categories', inplace=True, axis=1)
    df_merged = pd.merge(df_merged, categories, left_index=True, right_index=True)
    df_merged.drop('related', inplace=True, axis=1)
    # df_merged.plot(subplots=True, layout=(7,6), linestyle="None", marker='+') # look at no. unique values in each col
    # pd.Series(categories[column]).str.extract('\-(.*)').astype('int')
    return df_merged


def clean_data(df):
    """
    Remove unhelpful columens from dataframe (i.e. duplicates or columns which are multiclass rather than binary)
    :param df: dataframe that might whose columns need to be cleaned
    :return: df: dataframe with unhelpful columns removed
    """
    df = df.drop_duplicates()  # drop the duplicated rows
    return df


def save_data(df, database_filename):
    """
    Save merged dataframe in sqllite database
    :param df: dataframe to be stored
    :param df: filde location at which to store database
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}', echo=False)
    df.to_sql('tempTable', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    """
    load csv messages and categories and store them in the dataframe without columns that would not help with
    message categorization. Store the cleaned dataframe in sqllite for futher analsysis in other scripts.
    """
    main()

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df_merged = pd.merge(df_messages, df_categories, left_index=True, right_index=True)
    categories = df_merged.categories.str.split(';', expand=True)
    category_colnames = categories.loc[0].values
    category_colnames = list(map(lambda x: x[0:-2], category_colnames)) # use the string part of category values to get column names
    categories.columns = category_colnames  # add names to the category columns


    for column in categories:
        # set each value to be the last character of the string
        # categories[column] =
        categories[column] = categories[column].str.extract(r'\-(.*)').astype('int')  # extract first number found in each category

    categories.set_index(df_merged['id_x'])
    df_merged.drop('categories', inplace=True, axis=1)
    df_merged = pd.merge(df_merged, categories, left_index=True, right_index=True)

    #pd.Series(categories[column]).str.extract('\-(.*)').astype('int')
    return df_merged


def clean_data(df):
    df = df.drop_duplicates()  # drop the duplicated rows
    return df


def save_data(df, database_filename):
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
    main()
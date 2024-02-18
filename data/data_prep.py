import sys
import pandas as pd
import numpy as np

##############################################################################

def load_data(interactions_filepath, articles_filepath):
    '''Load data from csv
    
    INPUT
        interactions_filepath: path of the articles_filepath.csv file that 
                                needs to be imported
        articles_filepath: path of the articles_community.csv.csv file that 
                            needs to be imported
    
    OUTPUT
        returns datframes of the data provided
    '''
    
    # Load dataframes
    interactions = pd.read_csv(interactions_filepath)
    articles = pd.read_csv(articles_filepath)
    del interactions['Unnamed: 0']
    del articles['Unnamed: 0']
    
    return interactions, articles

##############################################################################

def email_mapper(df):
    ''' Map the user email to a user_id column
    '''
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

##############################################################################

def clean_data(interactions, articles):
    ''' Clean the merged dataframe
    
    INPUT
        df: The preprocessed dataframe
    
    OUTPUT
        df (DataFrame): Cleaned database of merged messages and categories
    '''
    # Convert 'article_id' column to int
    interactions['article_id'] = interactions['article_id'].astype('int')
    
    # Drop duplicate articles
    articles = articles.drop_duplicates('article_id')

    # BackFill missing descriptions and body in df_content
    articles = articles.fillna(method='bfill', axis=1)
    
    # Map the user email to a user_id column and remove the email column
    email_encoded = email_mapper(interactions)
    del interactions['email']
    interactions['user_id'] = email_encoded
    
    # Create 'content' column to use in content-based recommendation
    articles['content'] = articles['doc_full_name'] + ' ' + articles['doc_description']
    
    return interactions, articles
    
##############################################################################

def save_data(interactions, articles, data_dir):
    interactions.to_feather(data_dir + 'interactions.feather')
    articles.to_feather(data_dir + 'articles.feather')

##############################################################################
    
def main():
    if len(sys.argv) == 4:

        interactions_filepath, articles_filepath, data_dir = sys.argv[1:]

        print(
            'Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(
                interactions_filepath, articles_filepath
            )
        )
        interactions, articles = load_data(interactions_filepath, articles_filepath)

        print('Cleaning data...')
        interactions, articles = clean_data(interactions, articles)

        print('Saving data...\n')
        save_data(interactions, articles, data_dir)

        print('Cleaned data saved to feather files!')

    else:
        print(
            'Provide the filepaths of the interactions and articles '
            'datasets as the first and second argument respectively. '
            'Provide the save directory filepath as the third argument.'
            '\n\nExample: python process_data.py '
            'user-item-interactions.csv articles_community.csv data/'
        )


if __name__ == "__main__":
    main()
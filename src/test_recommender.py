import sys
import pandas as pd
from recommender import ArticleRecommender

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
    interactions = pd.read_feather(interactions_filepath)
    articles = pd.read_feather(articles_filepath)
    
    return interactions, articles

def main():
    if len(sys.argv) == 4:
        interactions_filepath, articles_filepath, user_id = sys.argv[1:]
        
        # Load dataframes
        interactions, articles = load_data(interactions_filepath, articles_filepath)
        
        # Initialize the recommender
        recommender = ArticleRecommender(articles, interactions)

        # Get recommendations for a user
        recommendations = recommender.recommend_articles(int(user_id))
        print(recommendations)
    else:
        print(
            'Provide the filepaths of the interactions and articles '
            'feather files as the first and second argument respectively. '
            'Provide the user_id to test as the third argument.'
            '\n\nExample: python test_recommender.py '
            'interactions.feather articles.feather 8'
        )


if __name__ == "__main__":
    main()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ArticleRecommender:
    def __init__(self, articles_df, interactions_df):
        """
        Initialize the ArticleRecommender with the articles and interactions DataFrames.

        Args:
        articles_df (pd.DataFrame): DataFrame containing articles data.
        interactions_df (pd.DataFrame): DataFrame containing user-article interactions.
        """
        self.articles_df = articles_df
        self.interactions_df = interactions_df
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.create_article_profiles()

    def create_article_profiles(self):
        """
        Create article profiles using TF-IDF based on titles and descriptions.
        """

        # Initialize a TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Fit and transform the articles to create TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.articles_df['content'])

    def recommend_articles(self, user_id, top_n=5):
        """
        Recommend articles for a given user considering articles they've already read.
        """
        # Find articles read by the user
        read_articles = self.interactions_df[self.interactions_df['user_id'] == user_id]['article_id'].unique()

        # Get Indices for computing cosine similarity
        read_articles_indices = self.articles_df[self.articles_df['article_id'].isin(read_articles)]['article_id'].tolist()

        # If the user has not read any articles, return top N popular articles
        if len(read_articles_indices) == 0:
            return self.get_top_articles(top_n)

        # Compute similarity of read articles with all articles
        similarity = cosine_similarity(self.tfidf_matrix[read_articles_indices], self.tfidf_matrix)

        # Average the similarities
        avg_similarity = similarity.mean(axis=0)

        # Recommend top N articles not read by the user
        self.articles_df['similarity'] = avg_similarity
        recommendations = self.articles_df[~self.articles_df['article_id'].isin(read_articles)].nlargest(top_n, 'similarity')

        return recommendations[['article_id', 'doc_full_name']]

    def get_top_articles(self, top_n):
        """
        Return top N popular articles based on the number of interactions.

        Args:
        top_n (int): Number of top articles to return.
        """
        
        top_articles_names = self.interactions_df.groupby('title')['user_id'].count().sort_values(ascending=False)[:top_n].index
        top_articles_ids = self.interactions_df.groupby('article_id')['user_id'].count().sort_values(ascending=False)[:top_n].index

        recommendations = pd.DataFrame({'article_id': top_articles_ids, 'doc_full_name': top_articles_names})
        
        return recommendations  
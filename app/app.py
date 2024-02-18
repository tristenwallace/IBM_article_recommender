import sys
sys.path.append('../src/')

from flask import Flask
from flask import render_template, request
import pandas as pd
from recommender import ArticleRecommender
from img_search import search_image


app = Flask(__name__)

# load data
interactions = pd.read_feather('../data/interactions.feather')
articles = pd.read_feather('../data/articles.feather')

# Initialize the recommender
recommender = ArticleRecommender(articles, interactions)


@app.route("/")
@app.route('/home')
def home():
    
    return render_template("pages/home.html")

@app.route("/go")
def response():
    # save user input in query
    user_id = request.args.get('query', '') 
    
    # Defaults user_id to 1 if no input is given
    if user_id ==  '':
        user_id = 1
    
    # number of articles to return 
    top_n = 2
    
    # use model to predict classification for query
    rec_article_ids = recommender.recommend_articles(int(user_id), top_n=top_n)['article_id'].values
    rec_articles = articles[articles['article_id'].isin(rec_article_ids)]
    rec_articles['img'] = rec_articles.doc_full_name.apply(lambda title: search_image(title, width=600, height=400))
    
    # Get df of read articles for creating cards in go.html
    read_article_ids = interactions[interactions['user_id'] == int(user_id)]['article_id'].unique()
    read_articles = articles[articles['article_id'].isin(read_article_ids)][:top_n]
    read_articles['img'] = read_articles.doc_full_name.apply(lambda title: search_image(title, width=600, height=400))

    
    return render_template("pages/go.html",
                            user_id=user_id,
                            read_articles=read_articles,
                            rec_articles=rec_articles)

@app.route("/about")
def about():
    return render_template("pages/about.html")
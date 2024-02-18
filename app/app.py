from flask import Flask
from flask import render_template, request
import sys
sys.path.append('../src/')
from recommender import ArticleRecommender
import pandas as pd

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
    
    # use model to predict classification for query
    read_articles = interactions[interactions['user_id'] == int(user_id)]['article_id'].unique()
    rec_articles = recommender.recommend_articles(int(user_id))['article_id']
    
    return render_template("pages/go.html",
                            user_id=user_id,
                            read_articles=read_articles,
                            rec_articles=rec_articles)

@app.route("/about")
def about():
    return render_template("pages/about.html")
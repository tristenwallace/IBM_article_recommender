# Disaster Response Message Classification App

# Project Overview

This project aims to build and deploy an article recommendation system that suggests relevant articles to users based on their reading history. It leverages Natural Language Processing (NLP) techniques to analyze article content and user interactions to generate personalized recommendations. The final application is deployed on Heroku, providing a web interface for users to interact with the recommendation system.

## File Descriptions
```
- app/
  |- app.py                   # Flask application entry point
  |- templates/               # HTML templates for the web interface
  |- img_search.py            # Module to fetch relevant images for articles
  |- recommender.py           # Recommendation engine implementation

- data/
  |- articles.feather         # Dataset containing article information
  |- interactions.feather     # Dataset containing user-article interactions

- notebooks/
  |- eda.ipynb                # Jupyter notebook for exploring data and models

- .env                        # Environment variables for local development (not to be committed)

- Procfile                    # Specifies commands for Heroku app startup

- requirements.txt            # Lists all dependencies necessary for the project
```

## Installations

To set up this project locally, follow these steps:

1. Fork and clone the repository to your local machine:

2. Create virtual environment via conda or venv

```
- On macOS/Linux:

  python3 -m venv venv
  source venv/bin/activate

- On Windows:

  python -m venv venv
  .\venv\Scripts\activate

```
3. Install the required packages:

```
pip install -r requirements.txt
```

## Instructions

1. Run the following commands in the project's root directory to set up your data files.
    
```
python data/data_prep.py user-item-interactions.csv articles_community.csv data/
```

2. Set up the local environment variables by creating a `.env` file in the project root with the necessary configurations.

3. To run the project locally:

1. Make sure you're in the project app directory and the virtual environment is activated. Then start the Flask application:

```
flask run
```

3. Go to

```
http://127.0.0.1:5000
```

## Results

The application is deployed on Heroku and can be accessed at [ibm-article-recommender-80fc57dd9efa.herokuapp.com](ibm-article-recommender-80fc57dd9efa.herokuapp.com). The deployed version utilizes the same recommendation logic and provides a user-friendly interface for interacting with the recommendation system.

## Tools Used

- **Flask**: Web framework for building the web interface
- **Bootstrap**: Front-end framework for designing responsive and mobile-first web pages.
- **Pandas**: Data manipulation and analysis
- **NLTK**: Natural Language Processing tasks
- **Scikit-learn**: Machine learning algorithms for recommendation
- **Recommendation Models**: Various algorithms (e.g., content-based filtering, collaborative filtering) used to analyze user behavior and content features to generate personalized article recommendations.
- **Heroku**: Cloud platform for deploying the application

## Future Considerations

- Improving the recommendation algorithm by incorporating more sophisticated machine learning models (such as SVD).
- Enhancing the user interface for a better user experience.
- Implementing additional features such as article tagging and categorization.
- Scaling the application to handle a larger number of users and articles.

## Resources
* [Udacity]((https://www.kaggle.com/datasets/sidharth178/disaster-response-messages?select=disaster_messages.csv)) for the IBM datasets


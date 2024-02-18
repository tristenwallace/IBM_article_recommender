import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from config import unsplash_access_key
from config import placeholder_images

def extract_keywords(title):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(title)
    words = [word for word in words if word.lower() not in stop_words and word.isalnum()]

    tagged = nltk.pos_tag(words)
    keywords = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ')]
    print(' '.join(keywords))
    return ' '.join(keywords)


def search_image(query, width=None, height=None):
    # Replace 'YOUR_ACCESS_KEY' with your actual Unsplash access key
    access_key = unsplash_access_key
    url = 'https://api.unsplash.com/search/photos'
    params = {
        'query': extract_keywords(query),
        'client_id': access_key,
        'page': 1,
        'orientation': 'landscape',
        'per_page': 1  # Assuming you want only one image per query
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

        # Assuming the first image in the results is what you want
        results = response.json()['results']
        if results:
            image_url = results[0]['urls']['regular']
            
            # Append the dimension parameters to the URL
            image_url += f"&w={width}"
            if height:
                image_url += f"&h={height}"
            return image_url
        else:
            return np.random.choice(placeholder_images)  # No results found
    except requests.RequestException as e:
        print(f'Error during requests to {url} : {str(e)}')
        return np.random.choice(placeholder_images)
import requests
import numpy as np
import os

# Set NLTK_DATA environment variable to point to the local nltk_data directory
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


placeholder_images = [
    'https://images.unsplash.com/photo-1525338078858-d762b5e32f2c?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'https://images.unsplash.com/photo-1696258686454-60082b2c33e2?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8YWl8ZW58MHx8MHx8fDA%3D',
    'https://images.unsplash.com/photo-1677756119517-756a188d2d94?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8YWl8ZW58MHx8MHx8fDA%3D',
    'https://images.unsplash.com/photo-1495055154266-57bbdeada43e?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTh8fGFpfGVufDB8fDB8fHww',
    'https://images.unsplash.com/photo-1674027444485-cec3da58eef4?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'https://images.unsplash.com/photo-1666597107756-ef489e9f1f09?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
    'https://images.unsplash.com/photo-1684369175809-f9642140a1bd?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NTl8fGFpfGVufDB8fDB8fHww',
    'https://images.unsplash.com/photo-1670163426610-69cdc930f4e0?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTA2fHxhaXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1692607334827-4da64dcf2221?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTA3fHxhaXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1599941662219-f95e5d125ce7?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTE0fHxhaXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1677442135131-4d7c123aef1c?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTM0fHxhaXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1676411237170-ddca6e4c158a?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTMyfHxhaXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1523961131990-5ea7c61b2107?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8ZGF0YXxlbnwwfHwwfHx8MA%3D%3D',
    'https://images.unsplash.com/photo-1495592822108-9e6261896da8?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fGRhdGF8ZW58MHx8MHx8fDA%3D'
]

def extract_keywords(title):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(title)
    words = [word for word in words if word.lower() not in stop_words and word.isalnum()]

    tagged = nltk.pos_tag(words)
    keywords = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ')]

    return ' '.join(keywords)


def search_image(query, width=None, height=None):
    # Replace 'YOUR_ACCESS_KEY' with your actual Unsplash access key
    access_key = os.environ.get('UNSPLASH_ACCESS_KEY')
    print(access_key)
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
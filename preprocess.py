import re
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words]

    return ' '.join(words)

test_df = pd.read_parquet(r'C:\Users\khilr\Desktop\IS5126FinalProject\ag_news\data\test-00000-of-00001.parquet')
train_df = pd.read_parquet(r'C:\Users\khilr\Desktop\IS5126FinalProject\ag_news\data\train-00000-of-00001.parquet')

train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)

print('Cleaning Complete')

print(train_df.head(10))
print(test_df.head(10))

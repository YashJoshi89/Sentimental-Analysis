#GIVE input.csv and pos.csv first as it contain the dataset
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
from textblob import TextBlob
import csv


csv_file_path = 'input.csv'
df = pd.read_csv(csv_file_path)


def scrape_article_data(url):
    try:

        response = requests.get(url)
        response.raise_for_status()


        soup = BeautifulSoup(response.text, 'html.parser')


        entry_title = soup.find('h1', class_='entry-title').get_text() if soup.find('h1', class_='entry-title') else None


        content_div = soup.find('div', class_='td-post-content tagdiv-type')


        article_text = content_div.get_text(separator=' ', strip=True) if content_div else None

        return entry_title, article_text
    except Exception as e:
        print(f"Error scraping content from {url}: {str(e)}")
        return None, None


df[['entry_title', 'scraped_content']] = df['URL'].apply(scrape_article_data).apply(pd.Series)


df['scraped_content'] = df['scraped_content'].str.replace('\n', ' ')


result_df = df[['URL_ID', 'entry_title', 'scraped_content']]
print(result_df)


df.to_csv('yashjoshi.csv')
df['scraped_content']=df['scraped_content'].astype(str)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['real']=df['scraped_content'].apply(transform_text)

df.to_csv('yashjoshi.csv')


df = pd.read_csv('yashjoshi.csv')


positive_words = set()
negative_words = set()

with open('pos.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        positive_words.add(row['positive_words'])
        negative_words.add(row['negative_words'])



def count_positive_negative_words(text):
    if isinstance(text, float):  # Check if the value is a float (NaN)
        return 0, 0


    text_str = str(text)

    analysis = TextBlob(text_str)


    words = analysis.words


    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)

    return positive_count, negative_count



df[['Positive Score', 'Negative Score']] = df['real'].apply(count_positive_negative_words).apply(pd.Series)

def polarity_score(row):
    return (row['Positive Score'] - row['Negative Score'])/((row['Positive Score'] + row['Negative Score'])+0.000001)
df['Polarity Score'] = df.apply(polarity_score, axis=1)

df['real']=df['real'].astype(str)
df['words_real']=df['real'].apply(lambda x:len(nltk.word_tokenize(x)))
df['sent_real']=df['real'].apply(lambda x:len(nltk.sent_tokenize(x)))

def subjective_score(row):
    return(row['Positive Score']+ row['Negative Score'])/(row['words_real']+0.000001)
df['Subjective Score']=df.apply(subjective_score,axis=1)

def avg_sent_len(row):
    return(row['words_real'])/(row['sent_real'])
df['avg_sent_len']=df.apply(avg_sent_len,axis=1)

def complex_words(text, complexity_threshold):
    words = text.split()
    complex_words = [word for word in words if len(word) > complexity_threshold]
    return len(complex_words)

complexity_threshold = 8

df['Complex Word'] = df['real'].apply(lambda x: complex_words(x, complexity_threshold))

def percentage_complex_word(row):
    return(row['Complex Word'])/(row['words_real'])
df['percentage complex word']=df.apply(percentage_complex_word,axis=1)

def fog_index(row):
    return 0.4*(row['avg_sent_len']+row['percentage complex word'])
df['fog index']=df.apply(fog_index,axis=1)

def count_syllables(word):
    vowels = "aeiouAEIOU"
    exception_suffixes = ["es", "ed"]


    for suffix in exception_suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]


    syllable_count = 0


    for i, char in enumerate(word):

        if char in vowels and (i == 0 or word[i - 1] not in vowels):
            syllable_count += 1

    return syllable_count

df['Syllable Count'] = df['real'].apply(count_syllables)

def avg_num_w_per_s(row):
    return (row['words_real'])/(row['sent_real'])
df['Avg num words per sentence']=df.apply(avg_num_w_per_s,axis=1)

import re

def count_personal_pronouns(text):

    pronoun_pattern = re.compile(r'\b(?:I|we|my|ours|us)\b', flags=re.IGNORECASE)


    matches = pronoun_pattern.findall(text)


    matches = [match for match in matches if match.lower() != 'us']


    return len(matches)
df['Personal Pronoun Count'] = df['real'].apply(count_personal_pronouns)

df['char']=df['real'].apply(len)
def avg_word_length(row):
    return(row['char'])/(row['words_real'])
df['Avg word length']=df.apply(avg_word_length,axis=1)

df.drop(columns=['Unnamed: 0','entry_title'],inplace = True)

df.to_csv('Outputdata.csv')
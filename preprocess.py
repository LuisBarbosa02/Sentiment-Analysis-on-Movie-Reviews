# Import libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up
nltk.download('stopwords')
nltk.download('punkt_tab')

# Pre-processing
def stopword_removal_and_stem_tokenizer(text):
    """
    A function to remove stop words (except negations), tokenize, and stem.

    :param text: Text to be pre-processed.
    :type text: str
    :return: A list with stop words removed and stemmed.
    :rtype: list
    """
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    stop_words.difference({'not', 'no', 'never', "n't"}) # Negative words should be kept, as they are important
    tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in tokens if w not in stop_words]

    # Stemming
    snowball = SnowballStemmer(language='english')
    stemmed_tokens = [snowball.stem(w) for w in filtered_tokens]

    return stemmed_tokens

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    tokenizer=stopword_removal_and_stem_tokenizer,
    preprocessor=None,
    lowercase=False,
    ngram_range=(1,2),
    max_features=50000
)
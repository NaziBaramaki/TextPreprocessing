import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer


# Download the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "This is a sample document. This document is a sample."

# Tokenize the text
tokens = word_tokenize(text.lower())  # Convert to lowercase

# Define punctuation and stop words
punctuation = set(string.punctuation)
stop_words = set(stopwords.words('english'))

# Remove punctuation and stop words
cleaned_tokens = [word for word in tokens if word not in punctuation and word not in stop_words]

# Create a Bag of Words representation
bow = Counter(cleaned_tokens)
# print(bow)

# Creating Object of PorterStemmer
stemmer = PorterStemmer()

# Stemming words
stemmed_words = [stemmer.stem(word) for word in tokens]

print(stemmed_words)
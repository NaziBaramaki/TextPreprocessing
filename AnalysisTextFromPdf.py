import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer



def ReadPdf(filePath):
    # Open the PDF file
    with open(filePath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # Get the number of pages
        num_pages = len(reader.pages)
        print(f'Number of pages: {num_pages}')

        # Read and print text from each page
        for page in range(num_pages):
            text = reader.pages[page].extract_text()
            print(f'Page {page + 1}:\n{text}\n')

    return text



# Function to read file and print sentences
def read_sentences_from_file(file_path):

    # Ensure you have the necessary NLTK resources
    nltk.download('punkt')

    text = ReadPdf(file_path)
    sentences = sent_tokenize(text)  # Split text into sentences
    return sentences


def processText(text, top_n=10):

    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert to lowercase

    # Define punctuation and stop words
    punctuation = set(string.punctuation)
    stop_words = set(stopwords.words('english'))

    # Remove punctuation and stop words, and ensure uniqueness
    unique_tokens = set(word for word in tokens if word not in punctuation and word not in stop_words)

    # Convert back to a list if needed (optional, as sets are unordered)
    cleaned_tokens = list(unique_tokens)

    tokenCount = Counter(cleaned_tokens)

    # Find the most important word
    # most_important_word = max(importance_scores, key=importance_scores.get)
    most_important_words = sorted(cleaned_tokens, reverse=True)[:top_n]

    return most_important_words

def findingStemmer(text):
    # Creating Object of PorterStemmer
    stemmer = PorterStemmer()

    # Stemming words
    stemmed_words = [stemmer.stem(word) for word in text]

    return(stemmed_words)


def FeaturExtracting(text):

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(text)

    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense().tolist()

    # Display TF-IDF scores for each document
    for i, doc in enumerate(dense):
        print(f"Document {i + 1}:")
        for j, score in enumerate(doc):
            if score > 0:
                print(f"  {feature_names[j]}: {score:.4f}")

    return tfidf_matrix

FilePass='c:/Users/Naziii/Desktop/Sector Map.pdf'

# Example usage
text = ReadPdf(FilePass)

important_words = processText(text, top_n=1000)
# print("Most Important Word:", important_words)

stemmed_words = findingStemmer(important_words)
# print(stemmed_words)

sentences= read_sentences_from_file(FilePass)

results = FeaturExtracting(sentences)
# print(results)




import easyocr
import pytesseract
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import BertTokenizer, BertModel
import spacy


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image
image = Image.open('c:/Users/Naziii/Desktop/paint2.jpg')
# Display the image
# image.show()

# Reading text
def extract_text_from_image(image_path):
    # img = Image.open(image_path)
    # reader = easyocr.Reader(['en'])  # Specify the languages you need
    # # Read text from an image
    # results = reader.readtextlang(img)

    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    # extracted_text = ' '.join([result[1] for result in results])
    return extracted_text

def process_with_bert(text, top_n=10):

    # Load spaCy English model for stop words
    nlp = spacy.load("en_core_web_sm")

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenizing Text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Get the outputs from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the last hidden states
    hidden_states = outputs.last_hidden_state

    # Calculate the mean of the hidden states for each token
    token_embeddings = hidden_states.mean(dim=1).squeeze()

    # Get the token IDs and convert them to words
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

    # Pair tokens with their embeddings
    word_embeddings = {token: token_embeddings[i].numpy() for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]']}

    # Process the text with spaCy to identify stop words
    doc = nlp(text)

    # Ensure the lengths match
    if len(tokens) != len(doc):
        print("Token length does not match document length. Adjusting indices.")
        min_length = min(len(tokens), len(doc))
    else:
        min_length = len(tokens)

    # Pair tokens with their embeddings, excluding stop words and special tokens
    word_embeddings = {}
    for i in range(min_length):
        token = tokens[i]
        if token not in ['[CLS]', '[SEP]'] and not doc[i].is_stop:
            word_embeddings[token] = hidden_states[0][i].numpy()  # Use embeddings directly

    # Calculate importance (here we simply use the magnitude of the embeddings)
    importance_scores = {word: (embedding**2).sum() ** 0.5 for word, embedding in word_embeddings.items()}

    # Find the most important word
    # most_important_word = max(importance_scores, key=importance_scores.get)
    most_important_words = sorted(importance_scores, key=importance_scores.get, reverse=True)[:top_n]

    return most_important_words

# Example usage
text = extract_text_from_image('c:/Users/Naziii/Desktop/paint2.jpg')
# important_word = process_with_bert(text)
important_words = process_with_bert(text, top_n=10)
print("Most Important Word:", important_words)
from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import contractions
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('./RNN_MOVIE_REVIEW.h5')

chat_words = {
    # Your chat words dictionary
}


def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


def remove_html_tags(text):
    return re.sub('<.*?>', '', text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def stopwords_removal(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def remove_emoji(text):
    return text.encode('ascii', 'ignore').decode('ascii')


def expand_contractions(text):
    return contractions.fix(text)


lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    word_list = word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predicate():
    input_text = request.form['review']
    print(input_text)
    # change lower case
    input_text = input_text.lower()
    # remove html tags
    input_text = remove_html_tags(input_text)
    # remove punctuation
    input_text = remove_punctuation(input_text)
    # chat conversion
    input_text = chat_conversion(input_text)
    # remove stopwords
    input_text = stopwords_removal(input_text)
    # remove emoji
    input_text = remove_emoji(input_text)
    # expand contractions
    input_text = expand_contractions(input_text)
    # lemmatization
    input_text = lemmatize_text(input_text)

    # Tokenization and padding
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, maxlen=250)

    prediction = model.predict(input_text)
    print(prediction)
    if prediction > 0.5:
        return 'Positive Review'
    else:
        return 'Negative Review'


if __name__ == '__main__':
    app.run(debug=True)

import re
import pickle
import os
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = ''

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)


def load_my_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except OSError as e:
        print(f"Error loading model: {e}")
        return None


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)


def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


def to_lower(text):
    return text.lower()


def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]


def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])


def clean_text(text):
    text = clean(text)
    text = is_special(text)
    text = to_lower(text)
    text = rem_stopwords(text)
    text = stem_txt(text)
    return text


model = load_my_model("./model/Movies_Review_Analysis_Model.keras")

if model is None:
    print("Model failed to load. Exiting.")
    exit(1)

with open('./model/Movies_Review_Analysis.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

max_sequence_length = 100


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.form['user_input'] == '':
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-red-800 font-bold'>Please enter a valid input</p>"

    try:
        user_input = request.form['user_input']
        user_input = clean_text(user_input)
        user_sequences = tokenizer.texts_to_sequences([user_input])
        user_padded = tf.keras.preprocessing.sequence.pad_sequences(
            user_sequences, maxlen=max_sequence_length)
        user_predictions = model.predict(user_padded)
        output = "Positive" if user_predictions[0] > 0.5 else "Negative"
        print(f'Predicted Class: {output}')
        print(user_predictions)
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-gray-800 font-bold'>" + output + "</p>"

    except Exception as e:
        print(f"Error predicting input: {e}")
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-red-800 font-bold'>Error analyzing the input</p>"


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

ps = PorterStemmer()
import pickle


def transform_text(features):
    features = features.lower()
    features = nltk.word_tokenize(features)
    y = []
    for i in features:
        if i.isalnum():
            y.append(i)
    features = y[:]
    y.clear()
    for i in features:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    features = y[:]
    y.clear()
    for i in features:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title('Email Spam Classifier')
input_sms = st.text_area('Enter the message')
if st.button('Predict'):
    transformed_text = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')



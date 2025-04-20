import streamlit as st

# Streamlit page config (must be first)
st.set_page_config(page_title="SMS Spam Detector", layout="centered")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import google.generativeai as genai

nltk.download('stopwords')

# Load and train model
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [w for w in text.split() if w not in set(stopwords.words('english'))]
        stemmed = [PorterStemmer().stem(w) for w in words]
        return ' '.join(stemmed)

    df['processed'] = df['message'].apply(preprocess)
    X = df['processed']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return model, vectorizer

model, vectorizer = load_and_train_model()

# Gemini AI config
genai.configure(api_key="AIzaSyA-DMRs8i1DP9Xfoxob4dF40Fx2zAS2sZQ")  # Replace with your key
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# App title
st.title("📩 SMS Spam Detector")

# Input
user_input = st.text_area("Enter an SMS message:", height=150)

# Preprocess input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in set(stopwords.words('english'))]
    stemmed = [PorterStemmer().stem(w) for w in words]
    return ' '.join(stemmed)

# Prediction logic
if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        # Local model output
        if pred == 1:
            st.error(f"🚨 **Spam** (Confidence: {prob[1]:.2%})")
        else:
            st.success(f"✅ **Not Spam** (Confidence: {prob[0]:.2%})")

        # Gemini validation
        with st.spinner("Validating with Gemini AI..."):
            prompt = f"""Classify this SMS as either 'Spam' or 'Not Spam':\n\n\"{user_input}\"\n\nOnly reply with 'Spam' or 'Not Spam'."""
            gemini_output = gemini_model.generate_content(prompt)
            response = gemini_output.text.strip().lower()

            if "spam" in response and "not spam" not in response:
                st.error("🔍 Gemini AI says: **Spam**")
            elif "not spam" in response:
                st.success("🔍 Gemini AI says: **Not Spam**")
            else:
                st.warning("Gemini AI couldn't classify the message.")

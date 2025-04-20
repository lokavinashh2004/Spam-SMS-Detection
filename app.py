import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

# Optional: Gemini API (install first: pip install google-generativeai)
import google.generativeai as genai

# Initialize Gemini (replace with your key)
GEMINI_API_KEY = "AIzaSyA-DMRs8i1DP9Xfoxob4dF40Fx2zAS2sZQ"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Download NLTK resources
nltk.download('stopwords')

# Page config
st.set_page_config(
    page_title="📱 SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput input {
        font-size: 16px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📱 SMS Spam Detection")
st.markdown("Using Machine Learning + Gemini AI to detect **Spam** or **Ham** messages.")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
    except:
        try:
            df = pd.read_csv('sms_spam_collection_dataset.csv', encoding='latin-1')
        except:
            st.error("Dataset not found. Please upload 'spam.csv' or 'sms_spam_collection_dataset.csv'.")
            return None
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

if df is not None:
    # Sidebar options
    st.sidebar.header("🔧 Settings")
    show_data = st.sidebar.checkbox("Show raw data")
    show_eda = st.sidebar.checkbox("Show EDA (Charts)")
    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
    random_state = st.sidebar.number_input("Random state", 0, 100, 42)

    # Preprocessing
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        stop_words = set(stopwords.words('english'))
        words = [w for w in text.split() if w not in stop_words]
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed)

    df['processed_message'] = df['message'].apply(preprocess_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Show raw data
    if show_data:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df[['label', 'message']])
        st.markdown(f"- Total Messages: {len(df)}")
        st.markdown(f"- Spam: {df['label'].sum()}")
        st.markdown(f"- Ham: {len(df) - df['label'].sum()}")

    # Split data
    X = df['processed_message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    # EDA
    if show_eda:
        st.subheader("📊 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Spam vs Ham")
            fig, ax = plt.subplots()
            df['label'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
            ax.set_xticklabels(['Ham', 'Spam'], rotation=0)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Message Length")
            df['length'] = df['message'].apply(len)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='length', hue='label', bins=40, ax=ax, multiple='stack', palette=['green', 'red'])
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Word Cloud (Ham)")
            ham_words = ' '.join(df[df['label'] == 0]['processed_message'])
            wc = WordCloud(width=800, height=400).generate(ham_words)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown("#### Word Cloud (Spam)")
            spam_words = ' '.join(df[df['label'] == 1]['processed_message'])
            wc = WordCloud(width=800, height=400).generate(spam_words)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    # Model performance
    st.subheader("📈 Model Performance")
    st.markdown(f"**Accuracy:** {accuracy:.2%}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Classification Report**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0))

    with col2:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        st.pyplot(fig)

    # Prediction
    st.subheader("🧪 Test a Message")
    user_input = st.text_area("Enter your SMS message:", "Congratulations! You won a free iPhone!")
    
    if st.button("Predict with ML"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a message.")
        else:
            processed_input = preprocess_text(user_input)
            input_vec = vectorizer.transform([processed_input])
            prediction = model.predict(input_vec)[0]
            prediction_prob = model.predict_proba(input_vec)[0]

            if prediction == 1:
                st.error(f"🚨 Spam detected! (Confidence: {prediction_prob[1]:.2%})")
            else:
                st.success(f"✅ It's Ham (Not Spam). (Confidence: {prediction_prob[0]:.2%})")

            st.bar_chart(pd.DataFrame({
                "Class": ["Ham", "Spam"],
                "Probability": [prediction_prob[0], prediction_prob[1]]
            }).set_index("Class"))

    # Gemini Classification
    if GEMINI_API_KEY:
        if st.button("Predict with Gemini AI"):
            st.markdown("🧠 Asking Gemini...")
            prompt = f"""Classify the following SMS as either Spam or Not Spam (Ham). 
Explain the reasoning briefly. Message: "{user_input}" """
            try:
                response = gemini_model.generate_content(prompt)
                st.info("📨 Gemini AI says:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error with Gemini API: {e}")

# Sidebar instructions
st.sidebar.markdown("""
---
### 📘 Instructions
1. Enter an SMS message.
2. Click on "Predict with ML" or "Predict with Gemini AI".
3. View results and charts from the sidebar.

### 📂 Dataset
[Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
""")

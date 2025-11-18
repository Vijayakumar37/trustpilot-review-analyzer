import re
import nltk
import joblib
import streamlit as st
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ========== NLTK setup ==========
# First run may need downloads; Streamlit Cloud will run this once at startup
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ========== Load model & vectorizer ==========
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("Review_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ========== Preprocessing functions (same as notebook) ==========
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

def get_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def categorize_sentiment(polarity: float) -> str:
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def predict_fake_or_genuine(text: str) -> str:
    cleaned = clean_text(text)
    processed = remove_stopwords(cleaned)
    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]
    return "Genuine" if pred == 0 else "Fake"

# ========== Streamlit UI ==========
st.title("Trustpilot Review Analyzer 🕵️‍♂️")
st.write("Enter a review and I’ll predict its **sentiment** and whether it looks **fake or genuine** (based on your trained model).")

user_input = st.text_area("Paste a review here:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        # Preprocess
        cleaned = clean_text(user_input)
        processed = remove_stopwords(cleaned)

        # Sentiment
        polarity = get_sentiment(processed)
        sentiment_label = categorize_sentiment(polarity)

        # Fake / Genuine
        fg_label = predict_fake_or_genuine(user_input)

        # Output
        st.subheader("Results")
        st.markdown(f"**Sentiment:** {sentiment_label}  (polarity = {polarity:.3f})")
        st.markdown(f"**Fake vs Genuine (model):** {fg_label}")

        with st.expander("Show cleaned text"):
            st.code(processed, language="text")

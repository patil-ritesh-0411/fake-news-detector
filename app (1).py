
import streamlit as st
import pickle
import re
import os
import gdown
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─── LOAD MODEL FROM GOOGLE DRIVE ───
@st.cache_resource
def load_models():
    if not os.path.exists("fake_news_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=19I5C6UAXWTsA4kLGJ0wrJlXS0llYQril", 
                      "fake_news_model.pkl", quiet=False)
    if not os.path.exists("tfidf_vectorizer.pkl"):
        gdown.download("https://drive.google.com/uc?id=1veVtaxDh8BsX7C78n2To8DUlB_bRjob9", 
                      "tfidf_vectorizer.pkl", quiet=False)
    
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_models()

# ─── CLEAN FUNCTION ───
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ─── PAGE CONFIG ───
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ─── CUSTOM CSS ───
st.markdown("""
<style>
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 16px !important;
    border: 2px solid #4a90e2 !important;
    border-radius: 10px !important;
}
.stButton > button {
    background-color: #4a90e2;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───
st.markdown("<h1 style=\'text-align:center;\'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style=\'text-align:center; color:gray;\'>Powered by Random Forest & NLP | IBM SkillsBuild Project</p>", unsafe_allow_html=True)
st.markdown("---")

# ─── INPUT ───
st.markdown("### 📝 Enter News Article or Headline:")
news = st.text_area("", height=200, placeholder="Paste your news text here...")

# ─── PREDICT ───
if st.button("🔍 Check News"):
    if news.strip() == "":
        st.warning("⚠️ Please enter some news text first!")
    else:
        with st.spinner("🔍 Analyzing..."):
            cleaned = clean_text(news)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            fake_pct = round(proba[0] * 100, 2)
            real_pct = round(proba[1] * 100, 2)

        st.markdown("---")
        if prediction == 1:
            st.success("✅ This news appears to be REAL!")
        else:
            st.error("❌ This news appears to be FAKE!")

        st.markdown("### 📊 Confidence Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Fake Probability", f"{fake_pct}%")
            st.progress(fake_pct / 100)
        with col2:
            st.metric("🟢 Real Probability", f"{real_pct}%")
            st.progress(real_pct / 100)

        st.markdown("---")
        st.caption("⚠️ This tool is for educational purposes only.")

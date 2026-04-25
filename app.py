
import streamlit as st
import joblib
import re

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_artifacts()

# -------------------- CLEAN FUNCTION --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# -------------------- UI --------------------
st.title("💬 Sentiment Analyzer (NLP)")
st.markdown("Enter a review and get sentiment with confidence")

user_input = st.text_area("✍️ Write your review:", height=150)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🚀 Analyze")

with col2:
    clear = st.button("🧹 Clear")

if clear:
    st.experimental_rerun()

# -------------------- PREDICTION --------------------
if analyze and user_input.strip() != "":

    text = clean_text(user_input)
    vector = tfidf.transform([text]).toarray()

    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    confidence = float(max(prob))

    st.divider()
    st.subheader("📊 Result")

    # RESULT DISPLAY
    if pred == 1:
        st.success(f"Positive 🙂  | Confidence: {confidence:.2f}")
    else:
        st.error(f"Negative 😡  | Confidence: {confidence:.2f}")

    # PROGRESS BAR
    st.progress(int(confidence * 100))

    # EXTRA INSIGHT
    st.subheader("🧠 Insight")

    if confidence < 0.6:
        st.warning("Model is unsure about this prediction.")
    elif confidence < 0.8:
        st.info("Moderate confidence prediction.")
    else:
        st.success("High confidence prediction.")

# -------------------- FOOTER --------------------
st.divider()
st.caption("Built with ❤️ using Machine Learning + NLP")

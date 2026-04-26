
import streamlit as st
import joblib

# ---------------- LOAD ----------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("nlp_pipeline.pkl")
    le = joblib.load("label_encoder.pkl")
    return pipeline, le

pipeline, le = load_model()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🤖", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    text-align:center;
    color:#4CAF50;
}
.sub-text {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="big-title">🤖 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Analyze text sentiment with confidence</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_area("✍️ Enter your review:", height=150)

# ---------------- BUTTON ----------------
if st.button("🚀 Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        pred = pipeline.predict([user_input])[0]
        prob = pipeline.predict_proba([user_input])[0]
        confidence = max(prob)

        label = le.inverse_transform([pred])[0]

        st.divider()
        st.subheader("📊 Result")

        # RESULT DISPLAY
        if label == "positive":
            st.success(f"😊 POSITIVE ({confidence:.2f})")
        else:
            st.error(f"😡 NEGATIVE ({confidence:.2f})")

        # CONFIDENCE BAR
        st.progress(int(confidence * 100))

        # INSIGHT
        if confidence < 0.6:
            st.warning("⚠️ Mixed or unclear sentiment")
        elif confidence < 0.8:
            st.info("ℹ️ Moderate confidence")
        else:
            st.success("🔥 High confidence")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with NLP + Machine Learning by You")

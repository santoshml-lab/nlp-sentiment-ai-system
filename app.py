
import streamlit as st
from transformers import pipeline

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.title {
    font-size:36px;
    font-weight:700;
    text-align:center;
    color:#4CAF50;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🤖 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by BERT (Transformers)</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_area("✍️ Enter your review:", height=150)

# ---------------- BUTTON ----------------
if st.button("🚀 Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        result = classifier(user_input)[0]
        label = result['label']
        confidence = result['score']

        st.divider()
        st.subheader("📊 Result")

        # RESULT DISPLAY
        if label == "POSITIVE":
            st.success(f"😊 POSITIVE ({confidence:.2f})")
            st.write("Glad you liked it! 🎉")
        else:
            st.error(f"😡 NEGATIVE ({confidence:.2f})")
            st.write("Sorry to hear that 😔")

        # CONFIDENCE BAR
        st.progress(int(confidence * 100))

        # SMART INSIGHT
        if confidence < 0.6:
            st.warning("⚠️ Model is unsure (ambiguous text)")
        elif confidence < 0.8:
            st.info("ℹ️ Moderate confidence")
        else:
            st.success("🔥 High confidence")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built using Transformers (BERT)")

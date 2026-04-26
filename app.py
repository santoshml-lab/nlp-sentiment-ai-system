
import streamlit as st
from transformers import pipeline
import random
import pandas as pd

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Sentiment Pro+", page_icon="🤖")

# ---------------- TITLE ----------------
st.title("🤖 AI Sentiment Pro+")
st.caption("Advanced AI Text Intelligence System")

# ---------------- INPUT ----------------
user_input = st.text_area("✍️ Enter your text:")

# ---------------- EMOJI SLIDER ----------------
emoji_level = st.slider("🎯 Mood Intensity", 0, 10, 5)

# ---------------- BUTTON ----------------
if st.button("🚀 Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter text")
    else:
        result = classifier(user_input)[0]
        label = result['label']
        confidence = result['score']

        # Save history
        st.session_state.history.append({
            "text": user_input,
            "label": label,
            "confidence": round(confidence,2)
        })

        # ---------------- RESULT ----------------
        st.subheader("📊 Result")

        if label == "POSITIVE":
            st.success(f"😊 POSITIVE ({confidence:.2f})")
        else:
            st.error(f"😡 NEGATIVE ({confidence:.2f})")

        st.progress(int(confidence * 100))

        # ---------------- MOOD RESPONSE ----------------
        if emoji_level > 7:
            st.write("🔥 Strong emotional tone detected!")
        elif emoji_level > 4:
            st.write("🙂 Moderate tone")
        else:
            st.write("😐 Neutral tone")

# ---------------- HISTORY ----------------
st.subheader("🕘 Analysis History")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # ---------------- CHART ----------------
    st.subheader("📈 Sentiment Distribution")

    chart_data = df["label"].value_counts()
    st.bar_chart(chart_data)

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Report",
        csv,
        "sentiment_report.csv",
        "text/csv"
    )

else:
    st.info("No history yet")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Built with AI | Premium Version")

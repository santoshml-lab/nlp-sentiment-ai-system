import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import plotly.express as px

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

clf = load_model()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sentiment AI Pro+", page_icon="🤖", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size:42px;
    font-weight:800;
    text-align:center;
    color:#4CAF50;
}
.sub {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
.card {
    background-color:#1e293b;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🤖 Sentiment AI Pro+</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Smart sentiment + analytics dashboard</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
col1, col2 = st.columns([2,1])

with col1:
    text = st.text_area("✍️ Enter text:", height=150)

with col2:
    st.info("💡 Tip:\nTry mixed or tricky sentences to test AI")

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze"):
    if text.strip() == "":
        st.warning("Enter some text")
    else:
        res = clf(text)[0]
        label = res["label"]
        conf = float(res["score"])

        # Save history
        st.session_state.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "label": label,
            "confidence": round(conf,2)
        })

        st.divider()

        # ---------------- RESULT CARDS ----------------
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Result")
            if label == "POSITIVE":
                st.success(f"😊 POSITIVE ({conf:.2f})")
            else:
                st.error(f"😡 NEGATIVE ({conf:.2f})")

            st.progress(int(conf * 100))

        with c2:
            st.subheader("🧠 Insight")
            if label == "POSITIVE":
                if conf > 0.85:
                    st.success("🔥 Strong Positive")
                elif conf > 0.65:
                    st.info("🙂 Moderate Positive")
                else:
                    st.warning("😐 Slight Positive")
            else:
                if conf > 0.85:
                    st.error("💥 Strong Negative")
                elif conf > 0.65:
                    st.warning("😕 Moderate Negative")
                else:
                    st.info("😐 Slight Negative")

# ---------------- HISTORY ----------------
st.divider()
st.subheader("🕘 History Dashboard")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df, use_container_width=True)

    # ---------------- PLOTLY CHART ----------------
    st.subheader("📊 Sentiment Distribution")

    fig = px.pie(
        df,
        names="label",
        title="Sentiment Breakdown",
        color="label",
        color_discrete_map={"POSITIVE":"green","NEGATIVE":"red"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- CONFIDENCE TREND ----------------
    st.subheader("📈 Confidence Trend")

    fig2 = px.line(
        df,
        x="time",
        y="confidence",
        markers=True,
        title="Confidence over Time"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Data", csv, "data.csv", "text/csv")

    # ---------------- CLEAR ----------------
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.success("History cleared!")

else:
    st.info("No data yet")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Premium AI Dashboard | Plotly Visuals")

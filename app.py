import streamlit as st
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 About the Model")
    st.markdown("""
    **Fake News Detection System** built using  
    **Machine Learning & NLP**
    
    ---
    ### 🔍 Algorithm Used
    - Logistic Regression  
    - TF-IDF Vectorization
    
    ---
    ### 📊 Dataset
    - Real & Fake news articles  
    - Text preprocessed using NLP techniques  
    
    ---
    ### ⚙️ Features
    - Text cleaning  
    - Stopword removal  
    - TF-IDF weighting  
    
    ---
    ### 🎯 Output
    - REAL or FAKE news  
    - Confidence score  
    
    ---
    ### 🛠 Tech Stack
    - Python  
    - Scikit-learn  
    - Streamlit  
    - NLP  
    
    ---
    ### 📌 Note
    This system predicts authenticity based on
    **patterns learned from historical data**.
    """)
    st.markdown("---")
    st.caption("Made by ❤️ using ML & AI")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* REMOVE TOP BAR */
header {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

.block-container {
    padding-top: 1rem !important;
}

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #1e3c72, #0f2027);
    background-attachment: fixed;
    color: white;
}

/* PARTICLES */
.particles {
    position: fixed;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    z-index: 0;
    overflow: hidden;
}

.particles span {
    position: absolute;
    width: 6px;
    height: 6px;
    background: rgba(255,255,255,0.25);
    border-radius: 50%;
    animation: float 15s linear infinite;
}

@keyframes float {
    from {transform: translateY(100vh);}
    to {transform: translateY(-10vh);}
}

/* CARD */
.card {
    position: relative;
    z-index: 2;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 40px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.6);
    max-width: 800px;
    margin: auto;
    margin-top: 60px;
}

/* TITLE */
.title h1 {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
}

.title p {
    font-size: 18px;
    text-align: center;
    opacity: 0.85;
}

/* TEXTAREA */
.stTextArea textarea {
    background: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: #0b132b;
    font-size: 18px;
    font-weight: bold;
    border-radius: 16px;
    padding: 14px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 30px rgba(79,172,254,0.9);
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 30px;
    font-size: 14px;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PARTICLES ----------------
st.markdown(
    "<div class='particles'>" +
    "".join([f"<span style='left:{i*5}%; animation-delay:{i}s'></span>" for i in range(20)]) +
    "</div>",
    unsafe_allow_html=True
)

# ---------------- MAIN UI ----------------
st.markdown("""
<div class="card">
    <div class="title">
        <h1>📰 Fake News Detector</h1>
        <p>Verify the authenticity of news articles using Artificial Intelligence</p>
    </div>
""", unsafe_allow_html=True)

news = st.text_area("📝 Paste News Article Here", height=220)

if st.button("🔍 Analyze News"):
    if news.strip():
        with st.spinner("Analyzing with AI..."):
            time.sleep(1)

        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        confidence = max(model.predict_proba(vec)[0]) * 100

        if pred[0] == 1:
            st.success("✅ The News is REAL")
        else:
            st.error("❌ The News is FAKE")

        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("⚠️ Please enter some news text.")

st.markdown("""
</div>
<div class="footer">
Built with ❤️ using Machine Learning, NLP & Streamlit
</div>
""", unsafe_allow_html=True)

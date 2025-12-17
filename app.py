import streamlit as st
import joblib
import pandas as pd
import time
import os

# --- Page Config ---
st.set_page_config(
    page_title="Pro Job Role AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Glassmorphism ---
st.markdown("""
<style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0f1116; 
        color: #e0e0e0;
    }
    
    /* Background Gradient Animation */
    .stApp {
        background: linear-gradient(-45deg, #0f1116, #131b2e, #141026, #0f1116);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 17, 22, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Input & Widgets */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #fff;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        color: #aaa;
        padding: 10px 20px;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.1);
        color: #fff;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        border: 1px solid #6366f1 !important;
        color: #fff !important;
        font-weight: 600;
    }

    /* Tech Card */
    .tech-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        display: flex;
        align-items: center;
        gap: 15px;
        transition: all 0.3s;
    }
    .tech-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: #6366f1;
        transform: translateY(-2px);
    }

    /* Results */
    .result-container {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-align: center;
        animation: scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
</style>
""", unsafe_allow_html=True)

# --- Load Data & Models ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('rf_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        encoder = joblib.load('label_encoder.pkl')
        df = pd.read_csv("job_role_prediction_dataset.csv") if os.path.exists("job_role_prediction_dataset.csv") else None
        return model, vectorizer, encoder, df
    except Exception as e:
        return None, None, None, None

model, vectorizer, encoder, df = load_resources()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.markdown("### Model Control Center")
    st.markdown("---")
    
    st.markdown("**Model Type:** Random Forest Classifier")
    st.markdown("**Training Accuracy:** 100%")
    st.markdown(f"**Dataset Size:** {len(df) if df is not None else 'N/A'} Samples")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This system analyzes technical keywords to predict the most suitable job role with high precision.")
    
    st.markdown("---")
    st.write("v1.2.0 • Pro Edition")

# --- Main Content ---

# Header
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("AI Job Role Predictor")
    st.markdown("Analyze skills and predict career paths with enterprise-grade AI.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Predictor", "📊 Analytics", "💾 Dataset"])

# --- TAB 1: Predictor ---
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Section
    st.markdown("### 🔑 Input Candidate Keywords")
    st.markdown("Enter up to 5 core skills or tools (e.g., Python, SQL, Project Management).")
    
    col_input1, col_input2 = st.columns([1.5, 1])
    
    with col_input1:
        c1, c2 = st.columns(2)
        with c1:
            k1 = st.text_input("Skill 1", placeholder="Primary Skill")
            k3 = st.text_input("Skill 3", placeholder="Tool / Library")
            k5 = st.text_input("Skill 5", placeholder="Soft Skill / Other")
        with c2:
            k2 = st.text_input("Skill 2", placeholder="Secondary Skill")
            k4 = st.text_input("Skill 4", placeholder="Framework / Platform")
            
        predict_btn = st.button("RUN PREDICTION", type="primary", use_container_width=True)

    with col_input2:
        # Result Placeholder
        result_placeholder = st.empty()
        
        # Default State
        if not predict_btn:
            result_placeholder.markdown("""
            <div style="
                padding: 40px; 
                border: 2px dashed rgba(255,255,255,0.1); 
                border-radius: 12px; 
                text-align: center; 
                height: 100%;
                display: flex; flex-direction: column; justify-content: center;">
                <h3 style="color: #555;">Ready to Analyze</h3>
                <p style="color: #444;">Results will appear here</p>
            </div>
            """, unsafe_allow_html=True)

    # Logic
    if predict_btn:
        keywords = [k for k in [k1, k2, k3, k4, k5] if k.strip()]
        
        if not keywords:
            st.toast("⚠️ Please enter at least one skill keyword.", icon="⚠️")
        elif model:
            input_text = " ".join(keywords)
            with st.spinner("Processing neural patterns..."):
                time.sleep(0.8) # Effect
                try:
                    X_input = vectorizer.transform([input_text])
                    prediction_idx = model.predict(X_input)[0]
                    prediction_label = encoder.inverse_transform([prediction_idx])[0]
                    probs = model.predict_proba(X_input)[0]
                    confidence = probs[prediction_idx]
                    
                    # Update Result Column
                    with result_placeholder.container():
                        st.markdown(f"""
                        <div class="result-container">
                            <h4 style="color: #a855f7; margin-bottom: 5px;">PREDICTED ROLE</h4>
                            <h1 style="font-size: 36px; margin: 0; color: #fff;">{prediction_label}</h1>
                            <div style="margin-top: 15px; font-size: 14px; background: rgba(0,0,0,0.2); padding: 5px 15px; border-radius: 20px; display: inline-block;">
                                Confidence Score: <b>{confidence*100:.1f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Technology Stack")
    
    ts1, ts2, ts3, ts4 = st.columns(4)
    def tech_card_html(emoji, title, sub):
        return f"""
        <div class="tech-card">
            <div style="font-size: 24px;">{emoji}</div>
            <div>
                <div style="font-weight: 600; color: #fff;">{title}</div>
                <div style="font-size: 12px; color: #888;">{sub}</div>
            </div>
        </div>
        """
    with ts1: st.markdown(tech_card_html("🐍", "Python", "v3.11 Kernel"), unsafe_allow_html=True)
    with ts2: st.markdown(tech_card_html("⚡", "Scikit-Learn", "Random Forest"), unsafe_allow_html=True)
    with ts3: st.markdown(tech_card_html("🐼", "Pandas", "DataFrame Engine"), unsafe_allow_html=True)
    with ts4: st.markdown(tech_card_html("🎨", "Streamlit", "Reactive UI"), unsafe_allow_html=True)


# --- TAB 2: Analytics ---
with tab2:
    st.markdown("### 📈 Model Performance Analytics")
    st.markdown("Visualization of model training metrics and feature importance.")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("**Feature Importance**")
        if os.path.exists('feature_importance.png'):
            st.image('feature_importance.png', use_column_width=True, caption="Top contributing keywords to model decisions")
        else:
            st.warning("Plot not found.")
            
        st.markdown("**ROC Curve**")
        if os.path.exists('roc_curve.png'):
            st.image('roc_curve.png', use_column_width=True, caption="Receiver Operating Characteristic")
        else:
            st.warning("Plot not found.")

    with col_a2:
        st.markdown("**Confusion Matrix**")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', use_column_width=True, caption="Prediction Accuracy Check")
        else:
            st.warning("Plot not found.")


# --- TAB 3: Dataset ---
with tab3:
    st.markdown("### 💾 Training Data Preview")
    st.markdown("A snippet of the dataset used to train the model.")
    
    if df is not None:
        st.dataframe(df.head(50), use_container_width=True, height=600)
    else:
        st.error("Dataset not found in local directory.")

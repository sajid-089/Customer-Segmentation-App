import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px  # Professional charts ke liye
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Analytics Pro | Customer Insights", layout="wide")

# --- CUSTOM CSS (Clean & Modern) ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1E3A8A; }
    .main-card { 
        background-color: white; 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE CLASS ---
class AnalyticsEngine:
    @st.cache_resource
    def load_assets(_self):
        m = joblib.load('customer_segmentation_model.pkl')
        s = joblib.load('data_scaler.pkl')
        c = joblib.load('cluster_map.pkl')
        d = pd.read_csv('Mall_Customers.csv')
        return m, s, c, d

engine = AnalyticsEngine()
model, scaler, cluster_info, df = engine.load_assets()

# --- HEADER ---
st.title("📊 Enterprise Customer Analytics")
st.caption("Strategic Segmentation Dashboard | Powered by Computational Math")

# --- LAYOUT: SIDEBAR & MAIN ---
with st.sidebar:
    st.header("📍 Input Parameters")
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)
    st.divider()
    st.write("### Project Metadata")
    st.write("**Model:** K-Means Clustering")
    st.write("**Accuracy:** Silhouette Optimized")

# --- MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["🎯 Prediction", "📈 Dataset Analysis"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Customer Profile")
        
        # Prediction Logic
        input_scaled = scaler.transform(np.array([[income, spending]]))
        cluster_id = model.predict(input_scaled)[0]
        res = cluster_info[cluster_id]
        
        st.metric("Detected Segment", res['label'])
        
        st.markdown(f"""
            <div style="padding:15px; border-radius:10px; border-left: 5px solid {res['color']}; background: {res['color']}15;">
                <h4 style="color:{res['color']}; margin:0;">{res['label']}</h4>
                <p style="font-size:12px; color:#555;">Segment characterized by income-to-spend ratio.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Segment Visualization")
        
        # Plotly Interactive Chart
        fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                         color=df['Cluster'].astype(str),
                         title="Market Map",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        
        # Add the User's point
        fig.add_scatter(x=[income], y=[spending], mode='markers', 
                        marker=dict(size=15, color='black', symbol='x'),
                        name='Current Input')
        
        fig.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Raw Data Insights")
    st.dataframe(df.style.background_gradient(subset=['Annual Income (k$)'], cmap='Greens'), use_container_width=True)

st.divider()
st.center = st.caption("© 2026 | Built for Data Science Portfolio")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import silhouette_score
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Strategy Hub", layout="wide", page_icon="📈")

# --- 2. LOAD ASSETS (Sahi Tareeqa) ---
@st.cache_resource
def load_all_assets():
    # Check if files exist to avoid crash
    files = ['customer_segmentation_model.pkl', 'data_scaler.pkl', 'cluster_map.pkl', 'Mall_Customers.csv']
    for f in files:
        if not os.path.exists(f):
            st.error(f"Missing File: {f}. Please upload it to GitHub.")
            return None, None, None, None
            
    m = joblib.load('customer_segmentation_model.pkl')
    s = joblib.load('data_scaler.pkl')
    c = joblib.load('cluster_map.pkl')
    d = pd.read_csv('Mall_Customers.csv')
    return m, s, c, d

model, scaler, cluster_info, df = load_all_assets()

# --- 3. UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-box { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION ---
if model is not None:
    # Adding cluster column to main df for visualizations
    X_vals = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    X_scaled = scaler.transform(X_vals)
    df['Cluster'] = model.predict(X_scaled)

    st.title("💎 Professional Customer Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Individual Prediction", "📊 Market Intelligence", "🚨 Anomaly Lab"])

    # --- TAB 1: INDIVIDUAL PREDICTION ---
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Customer Input")
            inc = st.number_input("Annual Income (k$)", 10, 200, 50)
            spd = st.slider("Spending Score (1-100)", 1, 100, 50)
            
            if st.button("Analyze Customer"):
                scaled_in = scaler.transform([[inc, spd]])
                pred = model.predict(scaled_in)[0]
                label = cluster_info[pred]['label']
                color = cluster_info[pred]['color']
                
                st.markdown(f"""
                    <div style="padding:20px; border-radius:10px; background:{color}22; border-left:10px solid {color};">
                        <h2 style="color:{color};">{label}</h2>
                        <p>This customer is classified as <b>{label}</b>.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Position in Market (3D)")
            fig3d = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                                  color='Cluster', opacity=0.7, template="plotly_white")
            st.plotly_chart(fig3d, use_container_width=True)

    # --- TAB 2: MARKET INTELLIGENCE ---
    with tab2:
        st.subheader("Accuracy & Correlation")
        m1, m2 = st.columns(2)
        
        # Silhouette Score Calculation
        sil_score = silhouette_score(X_scaled, df['Cluster'])
        m1.metric("Silhouette Score (Accuracy)", f"{sil_score:.2f}")
        m2.metric("Total Customers Analyzed", len(df))
        
        st.divider()
        st.write("### Feature Relationships")
        corr_fig = px.imshow(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(), text_auto=True)
        st.plotly_chart(corr_fig, use_container_width=True)

    # --- TAB 3: ANOMALY LAB ---
    with tab3:
        st.subheader("Outlier Detection")
        # Logic: Customers far from centroids
        dist = np.min(model.transform(X_scaled), axis=1)
        thresh = np.percentile(dist, 95)
        df['Anomaly'] = dist > thresh
        
        anom_df = df[df['Anomaly'] == True]
        st.warning(f"Detected {len(anom_df)} customers with unusual behavior (Outliers).")
        
        fig_anom = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                              color='Anomaly', color_discrete_map={True: 'red', False: 'blue'})
        st.plotly_chart(fig_anom, use_container_width=True)
        st.dataframe(anom_df)

st.divider()
st.caption("Computational Mathematics Dept | University of Karachi | Portfolio Project")
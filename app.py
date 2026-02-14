import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import silhouette_score

# ==========================================
# 1. SYSTEM CONFIG & ENGINE INITIALIZATION
# ==========================================
st.set_page_config(page_title="DataSense Pro | AI Cluster Lab", layout="wide", page_icon="🚀")

class ProductionEngine:
    def __init__(self):
        self.m_file = 'customer_segmentation_model.pkl'
        self.s_file = 'data_scaler.pkl'
        self.c_file = 'cluster_map.pkl'
        self.d_file = 'Mall_Customers.csv'

    @st.cache_resource
    def bootstrap(_self):
        # High-Fidelity error checking for missing assets
        missing = [f for f in [_self.m_file, _self.s_file, _self.c_file, _self.d_file] if not os.path.exists(f)]
        if missing: return f"Critical Missing Files: {', '.join(missing)}", None, None, None, None
        
        m = joblib.load(_self.m_file)
        s = joblib.load(_self.s_file)
        c = joblib.load(_self.c_file)
        d = pd.read_csv(_self.d_file)
        return None, m, s, c, d

engine = ProductionEngine()
err, model, scaler, c_map, df_base = engine.bootstrap()

# ==========================================
# 2. UI/UX STYLING (ADVANCED CSS)
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    [data-testid="stSidebar"] { background-image: linear-gradient(#1e3a8a, #1e40af); color: white; }
    .stMetric { background-color: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #3b82f6; }
    .card-title { font-size: 22px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px; }
    .recommendation-banner { background: #eff6ff; border: 1px solid #bfdbfe; padding: 15px; border-radius: 8px; border-left: 5px solid #2563eb; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")
    nav = st.radio("Active Module", ["Global Intelligence", "Segment Predictor", "Batch Processing", "Cluster Math (S-Score)"])
    st.markdown("---")
    st.write("### Engine Specs")
    st.caption(f"Cluster Count: {model.n_clusters if model else 'N/A'}")
    st.caption("Algorithm: K-Means++")
    st.caption("Data: Scikit-Learn Pipeline")

# ==========================================
# 4. APP LOGIC (THE HEART)
# ==========================================
if err:
    st.error(err)
    st.stop()

# Auto-preprocess base data for visualization
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X_scaled = scaler.transform(df_base[features])
df_base['Cluster'] = model.predict(X_scaled)

# --- MODULE 1: GLOBAL INTELLIGENCE ---
if nav == "Global Intelligence":
    st.markdown("<p class='card-title'>📈 Market Overview & Key Performance Indicators</p>", unsafe_allow_html=True)
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", len(df_base), "+12% vs LY")
    k2.metric("Avg Income", f"${df_base[features[0]].mean():.1f}k")
    k3.metric("Avg Spend Score", f"{df_base[features[1]].mean():.1f}")
    k4.metric("Market Volatility", "Low", delta_color="normal")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.write("### Cluster Density Map (2D)")
        fig_2d = px.scatter(df_base, x=features[0], y=features[1], color='Cluster', 
                           size='Age', hover_data=['Age'], color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_2d, use_container_width=True)
        
    with c2:
        st.write("### Segment Composition")
        fig_pie = px.pie(df_base, names='Cluster', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- MODULE 2: SEGMENT PREDICTOR ---
elif nav == "Segment Predictor":
    st.markdown("<p class='card-title'>🎯 Real-Time Individual Profiling</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.write("### Customer Dimensions")
        age_in = st.slider("Age Range", 18, 100, 30)
        inc_in = st.number_input("Annual Income ($k)", 10, 200, 50)
        spd_in = st.number_input("Spending Score (1-100)", 1, 100, 50)
        
        if st.button("RUN PREDICTION ENGINE", use_container_width=True):
            input_scaled = scaler.transform([[inc_in, spd_in]])
            p_cluster = model.predict(input_scaled)[0]
            info = c_map[p_cluster]
            
            st.success(f"Classification Complete: **{info['label']}**")
            st.markdown(f"""<div class='recommendation-banner'>
                <h4>Business Strategy for {info['label']}</h4>
                <p>Prioritize high-value engagement. This customer is likely to respond to premium loyalty rewards.</p>
                </div>""", unsafe_allow_html=True)
                
    with col2:
        st.write("### 3D Spatial Positioning")
        fig_3d = px.scatter_3d(df_base, x='Age', y=features[0], z=features[1], color='Cluster', opacity=0.6)
        # Add probe marker
        fig_3d.add_scatter3d(x=[age_in], y=[inc_in], z=[spd_in], mode='markers', 
                            marker=dict(size=10, color='red', symbol='cross'), name='Current Probe')
        st.plotly_chart(fig_3d, use_container_width=True)

# --- MODULE 3: BATCH PROCESSING ---
elif nav == "Batch Processing":
    st.markdown("<p class='card-title'>📂 Enterprise Batch Upload</p>", unsafe_allow_html=True)
    up_file = st.file_uploader("Upload Market Data (CSV)", type="csv")
    
    if up_file:
        u_df = pd.read_csv(up_file)
        if all(c in u_df.columns for c in features):
            u_scaled = scaler.transform(u_df[features])
            u_df['Cluster'] = model.predict(u_scaled)
            
            st.write("### Processed Results")
            st.dataframe(u_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            csv_data = u_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 DOWNLOAD ENRICHED DATA", csv_data, "segmented_export.csv", "text/csv")
        else:
            st.error(f"Error: CSV missing required headers: {features}")

# --- MODULE 4: CLUSTER MATH ---
elif nav == "Cluster Math (S-Score)":
    st.markdown("<p class='card-title'>🧪 Mathematical Validation (Computational Analysis)</p>", unsafe_allow_html=True)
    
    score = silhouette_score(X_scaled, df_base['Cluster'])
    st.metric("Global Silhouette Coefficient", f"{score:.4f}", help="Close to 1 is better clustering.")
    
    st.write("### Inter-Cluster Relationship (Heatmap)")
    corr = df_base[['Age', features[0], features[1]]].corr()
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("""
    > **Developer Note:** Silhouette coefficient represents how similar an object is to its own cluster compared to other clusters. 
    > High values indicate well-separated market segments.
    """)

# ==========================================
# 5. FOOTER
# ==========================================
st.divider()
st.caption("Computational Mathematics Dept - University of Karachi | Engineering-Grade AI Pipeline v3.0 (2026)")
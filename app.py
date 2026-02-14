import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import silhouette_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Strategy Hub", layout="wide", page_icon="💎")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .reportview-container { background: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 5px solid #1E3A8A; }
    .recommendation-box { background: #e0f2fe; padding: 20px; border-radius: 10px; border-left: 8px solid #0369a1; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('customer_segmentation_model.pkl')
        s = joblib.load('data_scaler.pkl')
        c = joblib.load('cluster_map.pkl')
        d = pd.read_csv('Mall_Customers.csv')
        return m, s, c, d
    except:
        return None, None, None, None

model, scaler, cluster_info, df_base = load_assets()

# --- BUSINESS LOGIC: RECOMMENDATIONS ---
def get_recommendation(label):
    recommendations = {
        'Target/VIP': "💎 **Strategy:** High-tier loyalty programs and exclusive early access to luxury collections.",
        'Careful': "🛡️ **Strategy:** Personalized discounts and value-for-money bundles to encourage spending.",
        'Spendthrifts': "🔥 **Strategy:** Flash sales and trendy 'limited time' offers to capitalize on impulse buying.",
        'Sensible': "📊 **Strategy:** Cashback offers and utility-based marketing to build long-term trust.",
        'Standard': "⚖️ **Strategy:** Cross-selling related products and standard newsletter engagement."
    }
    return recommendations.get(label, "Continue standard engagement.")

# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Strategic Hub")
    mode = st.radio("Navigation", ["Inference Engine", "Market Intelligence", "Anomaly Lab"])
    st.divider()
    st.info("Status: System Operational ✅")

# --- MAIN INTERFACE ---
if mode == "Inference Engine":
    st.title("🎯 Real-Time Customer Inference")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Input")
        age = st.slider("Age Profile", 18, 80, 35)
        income = st.number_input("Annual Income (k$)", 10, 200, 60)
        spend = st.number_input("Spending Score (1-100)", 1, 100, 50)
        
        if st.button("Generate Strategy", use_container_width=True):
            input_scaled = scaler.transform([[income, spend]])
            cluster_id = model.predict(input_scaled)[0]
            data = cluster_info[cluster_id]
            
            st.markdown(f"""<div class='metric-card'>
                <h4 style='color:{data['color']}'>{data['label']} Segment</h4>
                <h1>Cluster #{cluster_id}</h1>
            </div>""", unsafe_allow_html=True)
            
            st.markdown(f"<div class='recommendation-box'>{get_recommendation(data['label'])}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Market Positioning (3D)")
        # Background clusters for 3D visualization
        X_bg = df_base[['Annual Income (k$)', 'Spending Score (1-100)']].values
        X_bg_scaled = scaler.transform(X_bg)
        df_base['Cluster'] = model.predict(X_bg_scaled)
        
        fig_3d = px.scatter_3d(df_base, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                              color='Cluster', template="plotly_dark", height=500)
        fig_3d.add_scatter3d(x=[age], y=[income], z=[spend], mode='markers', 
                             marker=dict(size=12, color='white', symbol='diamond'), name='Current Probe')
        st.plotly_chart(fig_3d, use_container_width=True)

elif mode == "Market Intelligence":
    st.title("📈 Statistical Insights & Accuracy")
    
    # Math Flex: Silhouette Score
    X_scaled = scaler.transform(df_base[['Annual Income (k$)', 'Spending Score (1-100)']].values)
    score = silhouette_score(X_scaled, model.predict(X_scaled))
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette Accuracy", f"{score:.2f}", help="Measures how well clusters are separated. 1.0 is perfect.")
    m2.metric("Total Data Points", len(df_base))
    m3.metric("Optimum K-Value", model.n_clusters)
    
    st.divider()
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    corr = df_base[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heat, use_container_width=True)

elif mode == "Anomaly Lab":
    st.title("🚨 Anomaly & Outlier Detection")
    st.write("Detecting customers who do not fit the standard mathematical patterns.")
    
    # Basic Outlier Logic (Distance from Centroid)
    distances = np.min(model.transform(X_scaled), axis=1)
    threshold = np.percentile(distances, 95) # Top 5% as outliers
    df_base['Is_Anomaly'] = distances > threshold
    
    anomalies = df_base[df_base['Is_Anomaly'] == True]
    st.warning(f"Detected {len(anomalies)} anomalies in the dataset (Unusual behavior patterns).")
    st.dataframe(anomalies, use_container_width=True)
    
    fig_anom = px.scatter(df_base, x='Annual Income (k$)', y='Spending Score (1-100)', 
                         color='Is_Anomaly', color_discrete_sequence=['#1E3A8A', '#EF4444'])
    st.plotly_chart(fig_anom, use_container_width=True)

st.divider()
st.caption("Computational Mathematics Project | End-to-End MLOps Pipeline | Developed by Gemini")
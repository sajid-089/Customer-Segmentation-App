import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Cluster Analytics", layout="wide", page_icon="📈")

# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('customer_segmentation_model.pkl')
        scaler = joblib.load('data_scaler.pkl')
        cluster_info = joblib.load('cluster_map.pkl')
        return model, scaler, cluster_info
    except:
        return None, None, None

model, scaler, cluster_info = load_assets()

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Analytics Studio")
    input_mode = st.radio("Select Input Mode:", ["Manual Entry", "CSV Batch Upload"])
    st.divider()
    if model:
        st.success("Model Status: Online")
    else:
        st.error("Model Status: Offline")

# --- MAIN INTERFACE ---
st.title("🚀 Professional Customer Segmentation Dashboard")
st.write("Analyze customer behavior through Advanced K-Means Clustering.")

if input_mode == "Manual Entry":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Customer Profile")
        age = st.number_input("Age", 18, 100, 30)
        income = st.slider("Annual Income (k$)", 1, 200, 50)
        spending = st.slider("Spending Score (1-100)", 1, 100, 50)
        
        if st.button("Generate Insights", use_container_width=True):
            input_data = np.array([[income, spending]])
            scaled_data = scaler.transform(input_data)
            cluster_id = model.predict(scaled_data)[0]
            info = cluster_info[cluster_id]
            
            st.markdown(f"""
                <div style="background:{info['color']}22; border-left: 5px solid {info['color']}; padding:20px; border-radius:10px;">
                    <h3 style="color:{info['color']};">{info['label']}</h3>
                    <p>This customer is classified as <b>{info['label']}</b> based on the mathematical centroids of your market data.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 3D Cluster Topology")
        # Sample data visualization for context
        df_sample = pd.read_csv('Mall_Customers.csv')
        X_scaled = scaler.transform(df_sample[['Annual Income (k$)', 'Spending Score (1-100)']].values)
        df_sample['Cluster'] = model.predict(X_all_scaled if 'X_all_scaled' in locals() else X_scaled)
        
        fig = px.scatter_3d(df_sample, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                            color=df_sample['Cluster'].astype(str), 
                            title="3D Market Segmentation",
                            template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

elif input_mode == "CSV Batch Upload":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required = ['Annual Income (k$)', 'Spending Score (1-100)']
        if all(col in user_df.columns for col in required):
            # Processing
            X = user_df[required].values
            X_scaled = scaler.transform(X)
            user_df['Cluster'] = model.predict(X_scaled)
            
            # Dashboard Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers", len(user_df))
            m2.metric("Avg. Income", f"${user_df[required[0]].mean():.1f}k")
            m3.metric("Avg. Spend Score", f"{user_df[required[1]].mean():.1f}")
            
            # 2D Interactive Plot
            st.subheader("📈 Segment Distribution")
            fig_2d = px.scatter(user_df, x=required[0], y=required[1], color='Cluster',
                                hover_data=['Age'] if 'Age' in user_df.columns else None,
                                title="2D Segment Map")
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Data Preview
            with st.expander("View Processed Data"):
                st.dataframe(user_df, use_container_width=True)
                
            # Download Button
            csv = user_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")
        else:
            st.warning(f"CSV must contain these columns: {required}")

st.divider()
st.caption("Developed by Gemini AI for Computational Mathematics Portfolio | University of Karachi")
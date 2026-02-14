import streamlit as st
import joblib
import numpy as np
import os

# --- PROFESSIONAL UI CONFIG ---
st.set_page_config(
    page_title="Customer Analytics Pro",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a sleek look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2c3e50; color: white; }
    .reportview-container .main .footer { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

class DeploymentEngine:
    """Class to handle all ML operations safely."""
    def __init__(self):
        self.model_path = 'customer_segmentation_model.pkl'
        self.scaler_path = 'data_scaler.pkl'
        self.map_path = 'cluster_map.pkl'

    @st.cache_resource
    def load_resources(_self):
        """Loads and caches all ML assets."""
        if not all(os.path.exists(p) for p in [_self.model_path, _self.scaler_path, _self.map_path]):
            return None, None, None
        
        m = joblib.load(_self.model_path)
        s = joblib.load(_self.scaler_path)
        c = joblib.load(_self.map_path)
        return m, s, c

# --- APP EXECUTION ---
engine = DeploymentEngine()
model, scaler, cluster_info = engine.load_resources()

# Sidebar - Professional Inputs
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Control Panel")
    st.info("Enter customer metrics to classify segments.")
    
    income = st.number_input("Annual Income (k$)", 1, 200, 50, help="Customer's yearly earnings")
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)
    analyze_btn = st.button("RUN ANALYTICS")

# Main Dashboard
st.title("📊 Customer Segmentation Analytics")
st.markdown("---")

if model is None:
    st.error("❌ Critical Error: Model files not found. Please run the trainer script first.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💡 Input Summary")
        st.write(f"**Target Income:** ${income}k")
        st.write(f"**Target Spend Score:** {spending}/100")

    if analyze_btn:
        with col2:
            # Prediction Logic
            input_scaled = scaler.transform(np.array([[income, spending]]))
            cluster_id = model.predict(input_scaled)[0]
            data = cluster_info[cluster_id]

            # Professional Display
            st.subheader("🎯 Model Prediction")
            st.metric(label="Predicted Segment", value=data['label'])
            
            st.markdown(f"""
                <div style="border-left: 10px solid {data['color']}; padding: 20px; background-color: white; border-radius: 5px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color: {data['color']}; margin: 0;">{data['label']} Category</h3>
                    <p style="color: #666;">This customer has been classified based on spending patterns and income variance.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        with col2:
            st.info("Waiting for input... Click 'RUN ANALYTICS' to see the classification.")

st.markdown("---")
st.caption("Computational Mathematics Dept - University of Karachi | Engineering-Grade ML Deployment")
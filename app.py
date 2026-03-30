import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Weather AI Pro", page_icon="🌤️", layout="centered")

# Custom CSS for UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 20px; height: 3em;
        background-color: #00acee; color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #0078aa; }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px; padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Models Load Karein (Corrected Paths for Small Models)
@st.cache_resource
def load_all_assets():
    try:
        # Check karein agar models folder ke andar hain ya bahar
        model_path = 'models/' if os.path.exists('models') else ''
        
        temp_model = joblib.load(f'{model_path}weather_rf_model_small.pkl')
        rain_model = joblib.load(f'{model_path}rain_model_small.pkl')
        le = joblib.load(f'{model_path}state_encoder.pkl')
        return temp_model, rain_model, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model_temp, model_rain, le = load_all_assets()

# 3. Header Section
st.title("🌤️ AI Weather Analytics")
st.markdown("---")

# Model check alert
if model_temp is None or model_rain is None:
    st.warning("⚠️ Models load nahi ho paye. Check karein ki GitHub par 'models' folder mein '.pkl' files sahi naam se hain ya nahi.")
    st.info("Files required: weather_rf_model_small.pkl, rain_model_small.pkl, state_encoder.pkl")
    st.stop()

# 4. Input Section
with st.container():
    st.subheader("📍 Check Forecast")
    col_state, col_date = st.columns(2)
    
    with col_state:
        state_input = st.selectbox("Select State", options=le.classes_)
    
    with col_date:
        selected_date = st.date_input("Choose Date", datetime.now())

# Prediction Button
if st.button("🚀 Generate AI Forecast"):
    with st.spinner('AI is analyzing historical data...'):
        # Data Preprocessing
        state_encoded = le.transform([state_input])[0]
        month = selected_date.month
        day = selected_date.day
        year = selected_date.year
        
        # Input features logic (Average placeholders for simplicity)
        hour = 12
        humidity, pressure, wind, cloud = 55, 1010, 12, 30
        
        input_features = pd.DataFrame([[year, month, day, hour, state_encoded, humidity, pressure, wind, cloud]], 
                                     columns=['year', 'month', 'day', 'hour', 'State_Encoded', 'humidity', 'pressure', 'windspeedKmph', 'cloudcover'])
        
        # Predicting
        t_pred = model_temp.predict(input_features)[0]
        r_pred = model_rain.predict(input_features)[0]
        
        st.markdown("### 📊 Live Insights")
        
        # Result Layout
        res1, res2 = st.columns(2)
        
        with res1:
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="font-size:18px; color:#aaa;">Temperature</p>
                    <h1 style="color:#ff4b4b;">{t_pred:.1f}°C</h1>
                </div>
            """, unsafe_allow_html=True)
            
        with res2:
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="font-size:18px; color:#aaa;">Rainfall</p>
                    <h1 style="color:#00acee;">{r_pred:.2f} mm</h1>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Smart Feedback
        if r_pred > 0.5:
            st.warning("⚠️ **Precipitation Alert:** High chances of rain. Better to carry an umbrella!")
        else:
            st.success("✅ **Clear Skies:** Weather looks stable for outdoor activities.")

# 5. Footer
st.markdown("<br><p style='text-align: center; color: #555;'>Data Analytics Project | NIT Rourkela</p>", unsafe_allow_html=True)

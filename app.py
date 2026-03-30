import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# 1. Page Configuration & Custom CSS for Advanced UI
st.set_page_config(page_title="Weather AI Pro", page_icon="🌤️", layout="centered")

# Custom CSS for Background and Glassmorphism effect
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #00acee;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0078aa;
        border: none;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Models Load Karein
@st.cache_resource
def load_all_assets():
    try:
        temp_model = joblib.load('models/weather_rf_model.pkl')
        rain_model = joblib.load('models/rain_rf_model.pkl')
        le = joblib.load('models/state_encoder.pkl')
        return temp_model, rain_model, le
    except:
        return None, None, None

model_temp, model_rain, le = load_all_assets()

# 3. Header Section
st.title("🌤️ AI Weather Analytics")
st.markdown("---")

if model_temp is None:
    st.error("Error: Models folder check karein! .pkl files nahi mili.")
    st.stop()

# 4. Input Section - Minimalist UI
with st.container():
    st.subheader("📍 Check Forecast")
    
    col_state, col_date = st.columns(2)
    
    with col_state:
        state_input = st.selectbox("Select State", options=le.classes_)
    
    with col_date:
        # User ko Calendar dikhayenge (UX improvement)
        selected_date = st.date_input("Choose Date", datetime.now())

# Prediction Button
if st.button("🚀 Generate AI Forecast"):
    with st.spinner('AI is analyzing historical data...'):
        # Data Preprocessing
        state_encoded = le.transform([state_input])[0]
        month = selected_date.month
        day = selected_date.day
        year = selected_date.year
        
        # Logic values (Average placeholders)
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
        
        # Smart Feedback logic
        if r_pred > 0.5:
            st.warning("⚠️ **Precipitation Alert:** High chances of rain. Better to carry an umbrella!")
        else:
            st.success("✅ **Clear Skies:** Weather looks stable for outdoor activities.")

# 5. Footer
st.markdown("<br><p style='text-align: center; color: #555;'>Data Analytics Project | NIT Rourkela</p>", unsafe_allow_html=True)
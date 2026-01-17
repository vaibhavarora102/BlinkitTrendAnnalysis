import streamlit as st
import pandas as pd
import pickle
import datetime
import os
import joblib

# Page Config
st.set_page_config(page_title="Blinkit Sales Prediction", layout="wide")

# Title
st.title("🛍️ Blinkit Sales Prediction App")
st.markdown("Enter the daily metrics below to predict the **Total Sales Revenue**.")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("trained_models", "xgb1.joblib.dat")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model using the notebook first.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    # Input Form
    with st.form("prediction_form"):
        st.header("Daily Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            order_date = st.date_input("Order Date", datetime.date.today())
            price_effect_gap = st.number_input("Price Effect Gap (Actual - Expected Revenue)", value=0.0)
            stock_to_revenue_ratio = st.number_input("Stock to Revenue Ratio", value=0.0, min_value=0.0)
            
        with col2:
            revenue_growth_rate = st.number_input("Revenue Growth Rate", value=0.0)
            rolling_revenue_avg_7 = st.number_input("7-Day Rolling Revenue Avg", value=0.0)
            
        with col3:
            revenue_per_unit = st.number_input("Revenue Per Unit", value=0.0, min_value=0.0)
            avg_revenue_per_order = st.number_input("Avg Revenue Per Order", value=0.0, min_value=0.0)
            
        submit_button = st.form_submit_button("Predict Revenue")

    if submit_button:
        # Feature Engineering from Date
        day = order_date.day
        month = order_date.month
        day_of_week = order_date.weekday() # 0=Monday, 6=Sunday
        
        # Prepare Input Data
        # Order must match training: day, month, day_of_week, price_effect_gap, stock_to_revenue_ratio, 
        # revenue_growth_rate, rolling_revenue_avg_7, revenue_per_unit, avg_revenue_per_order
        
        input_data = pd.DataFrame({
            'day': [day],
            'month': [month],
            'day_of_week': [day_of_week],
            'price_effect_gap': [price_effect_gap],
            'stock_to_revenue_ratio': [stock_to_revenue_ratio],
            'revenue_growth_rate': [revenue_growth_rate],
            'rolling_revenue_avg_7': [rolling_revenue_avg_7],
            'revenue_per_unit': [revenue_per_unit],
            'avg_revenue_per_order': [avg_revenue_per_order]
        })
        
        try:
            prediction = model.predict(input_data)[0]
            
            st.success("### Prediction Successful!")
            st.metric(label="Predicted Total Sales", value=f"₹{prediction:,.2f}")
            
            # Show input summary
            with st.expander("See Input Details"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

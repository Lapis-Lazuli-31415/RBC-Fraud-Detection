import datetime
import json
import folium
import numpy as np
import streamlit as st
import joblib
import pandas as pd
from streamlit_folium import st_folium

# Load the model
pipeline = joblib.load("fraud_detection_xgb.pkl")

def transform_input(data: dict):
    df = pd.DataFrame([data])

    # gender_cate
    def gender_cate(gender: str) -> int:
        if gender == 'M':
            return 0
        elif gender == 'F':
            return 1
    df['gender_cate'] = df['gender'].apply(gender_cate)
    
    # datetime features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['month'] = df['trans_date_trans_time'].dt.month
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['minute'] = df['trans_date_trans_time'].dt.minute
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    
    # month_segment
    def month_segment(day: int) -> int:
        if day <= 10:
            return 0
        elif day <= 20:
            return 1
        else:
            return 2
    df['month_segment'] = df['month'].apply(month_segment)
    
    # city_size
    def city_size(city_pop: int) -> int:
        if city_pop < 50000:
            return 0
        elif city_pop < 200000:
            return 1
        elif city_pop < 500000:
            return 2
        elif city_pop < 1500000:
            return 3
        else:
            return 4
    df['city_size'] = df['city_pop'].apply(city_size)
    
    # amt_log
    df['amt_log'] = np.log1p(df['amt'])
    
    # dis_amt_prod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371  # Earth radius in km
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    df['dis_amt_prod'] = df.apply(lambda row: haversine_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']) * row['amt'],
        axis=1)
    
    # amt_hour_risk_prod
    def hour_risk(hour: int, minute: int) -> int:
        peak_hour = 2
        sigma = 2
        hour_decimal = hour + minute / 60
        risk = np.exp(-0.5 * ((hour_decimal - peak_hour) / sigma) ** 2)
        return risk
    df['amt_hour_risk_prod'] = df.apply(lambda row: row['amt'] * hour_risk(row['hour'], row['minute']), axis=1)
    
    # category_fraud_r (placeholder, replace with your mapping)
    with open("category_fraud_rates.json", "r") as f:
        category_fraud_rates = json.load(f)
    df['category_fraud_r'] = df['category'].map(category_fraud_rates).fillna(category_fraud_rates['__overall__'])
    
    # Final model columns
    return df[['gender_cate', 'month_segment', 'city_size', 'day_of_week', 
               'amt_hour_risk_prod', 'category_fraud_r', 'month', 
               'amt_log', 'hour', 'dis_amt_prod']]


# Streamlit App
st.title("Fraud Detection Predictor")


# User inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Gender", options=["M", "F"], index=0, horizontal=True)
    date_input = st.date_input("Transaction Date", value=datetime.date.today())
    time_input = st.time_input("Transaction Time", value=datetime.time(0, 0), step=60)
    trans_date = datetime.datetime.combine(date_input, time_input)
    city_pop = st.number_input("City Population", min_value=0)
    amt = st.number_input("Transaction Amount", min_value=0.0, step=0.01, format="%f")

with col2:
    category = st.selectbox("Category", ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel", "other"])
    lat = st.number_input("Cardholder Latitude", value=33.0, format="%f", min_value=-90.0, max_value=90.0, step=0.000001)
    long = st.number_input("Cardholder Longitude", value=-80.0, format="%f",min_value=-180.0, max_value=180.0, step=0.000001)
    merch_lat = st.number_input("Merchant Latitude", value=34.0, format="%f",min_value=-90.0, max_value=90.0, step=0.000001)
    merch_long = st.number_input("Merchant Longitude", value=-81.0, format="%f",min_value=-180.0, max_value=180.0, step=0.000001)

# Create map centered between the two points
st.markdown("### Transaction Map")
center_lat = (lat + merch_lat) / 2
center_lon = (long + merch_long) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add markers
folium.Marker([lat, long], popup="cardholder", icon=folium.Icon(color='blue', icon='credit-card')).add_to(m)
folium.Marker([merch_lat, merch_long], popup="merchant", icon=folium.Icon(color='red', icon='shopping-cart')).add_to(m)

# Draw line connecting them
folium.PolyLine(locations=[[lat, long], [merch_lat, merch_long]], color='green',weight=2,dash_array='5').add_to(m)

m.fit_bounds(bounds=[[lat, long], [merch_lat, merch_long]], padding=[50,50])

# Show map in Streamlit
st_folium(m, width=700, height=500)

if st.button("Predict Fraud"):
    raw_data = {
        "gender": gender,
        "trans_date_trans_time": trans_date,
        "city_pop": city_pop,
        "amt": amt,
        "category": category,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long
    }
    
    df_processed = transform_input(raw_data)
    
    # Prediction
    pred = pipeline.predict(df_processed)[0]
    proba = pipeline.predict_proba(df_processed)[:,1][0]
    
    if pred == 1:
        st.error(f"Fraudulent Transaction (Probability: {proba:.2%})")
    else:
        st.success(f"Legitimate Transaction (Probability of Fraud: {proba:.2%})")
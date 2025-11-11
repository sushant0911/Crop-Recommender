import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

# Set page configuration
st.set_page_config(
    page_title="Dynamic Crop Recommendation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌱"
)

# CSS with updated styling for dark theme
# 🌾 Farm-Themed UI with Light Background & Button Spacing
st.markdown("""
<style>
html, body, .main {
    background-color: #f9f8f3 !important;
    color: #2e382e !important;
    font-family: 'Poppins', sans-serif;
}

/* ===== Headings ===== */
h1 {
    color: #3e6e2c !important;
    text-align: center;
    font-weight: 700;
}
h2, h3, h4 {
    color: #4e7d2d !important;
    font-weight: 600;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background-color: #f1f8e9 !important;
    color: #2e382e !important;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
    color: #2e382e !important;
}
[data-testid="stSidebar"] a {
    color: #33691e !important;
    font-weight: 600;
}

/* ===== Tabs (Predict / History / Feedback) ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px; /* space between tabs */
}
.stTabs [data-baseweb="tab"] {
    background-color: #f1f8e9 !important;
    color: #3e4e27 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 25px !important; /* wider tabs */
}
.stTabs [aria-selected="true"] {
    background-color: #8bc34a !important;
    color: white !important;
}

/* ===== Buttons ===== */
.stButton {
    display: inline-block;
    margin-right: 15px; /* space between buttons */
}
.stButton>button {
    background: #8bc34a !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 25px !important; /* wider buttons */
    font-weight: 600 !important;
    font-size: 15px !important;
}
.stButton>button:hover {
    background: #689f38 !important;
    transform: scale(1.03);
}

/* ===== Cards ===== */
.card {
    background: #ffffff;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    padding: 20px 25px;
    margin-bottom: 20px;
    border-left: 5px solid #9ccc65;
}

/* ===== Inputs ===== */
.stNumberInput label,
.stTextInput label,
.stSelectbox label,
.stTextArea label,
.stSlider label {
    color: #355e2e !important;
    font-weight: 600 !important;
}
.stNumberInput input,
.stTextInput input,
.stSelectbox select,
textarea {
    background-color: #fcfff7 !important;
    color: #2e382e !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.6rem !important;
}
.stTextInput>div>div>input::placeholder {
    color: #888 !important;
}

/* ===== Tables ===== */
.custom-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #ffffff;
    border-radius: 10px;
    overflow: hidden;
}
.custom-table th {
    background-color: #c5e1a5;
    color: #2e382e;
    font-weight: 700;
    padding: 12px;
    text-align: left;
}
.custom-table td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
}

/* ===== Feedback Box ===== */
.feedback-success {
    background: #e8f5e9;
    border-left: 5px solid #81c784;
    color: #2e7d32;
    padding: 10px 15px;
    border-radius: 6px;
    margin-top: 10px;
}
            
.stMainBlockContainer  {
    background-color: #f9f8f3 !important;
}
            
.st-emotion-cache-14vh5up {
    background-color: #8bc34a !important;            
}

.st-dw {
    border-top-width: 1px;
    background-color: #fcfff7 !important;
    color: grey
}        
.st-dg {
    background-color: #fcfff7; 
}        
.st-bw{
    border: 0.1px solid #fcfff7 !important;            
}                     
.st-emotion-cache-9gx57n {
    display: flex;
    flex-flow: row;
    -webkit-box-align: center;
    align-items: center;
    height: 2.5rem;
    border-width: 0.1px;
    border-style: solid;
    border-color: #fcfff7;
    transition-duration: 200ms;
    transition-property: border;
    transition-timing-function: 
cubic-bezier(0.2, 0.8, 0.4, 1);
    border-radius: 0.5rem;
    overflow: hidden;
}
.st-fa {
    transition-duration: 400ms;
    background-color: transparent;
}
.st-emotion-cache-1nhqhy2 {
    margin: 0px;
    border: none;
    height: 100%;
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    width: 2rem;
    -webkit-box-pack: center;
    justify-content: center;
    color: rgb(250, 250, 250);
    transition: color 300ms, backgroundColor 300ms;
    background-color: #8bc34a;
}     

.st-emotion-cache-1vo6xi6 {
    width: 100%;
    height: auto;
    max-width: 100%;
    min-width: 1rem;
    position: relative;
    overflow: visible;
    color: grey;
}
.st-emotion-cache-11mwrlk {
    width: 100%;
    color: grey;
    border-spacing: 0px;
}   

.st-c2 {
    background-color: transparent !important;
}           
            
/* ===== Images ===== */
img {
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Sidebar
with st.sidebar:
    st.image("agri_logo.png", use_container_width=True)
    st.markdown("<h2 style='text-align: center;'>Crop Recommendation System</h2>", unsafe_allow_html=True)
    st.markdown("""
    This app uses a Random Forest model to recommend the best crops
    based on soil nutrients and weather conditions.
    """)
    st.markdown("---")
    st.markdown("**Developed by**: Team 8848 (Sushant Gupta, Sumit Gupta & Roshan Prasad)")

# Load model and scaler
try:
    rdf_clf = joblib.load('RDF_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found.")
    st.stop()

# Load crop descriptions
try:
    df_desc = pd.read_csv('Crop_Desc.csv', sep=';', encoding='utf-8')
    required_columns = [
        'label', 'image', 'N_min', 'N_max',
        'P_min', 'P_max', 'K_min', 'K_max',
        'pH_min', 'pH_max'
    ]
    missing_columns = [col for col in required_columns if col not in df_desc.columns]
    if missing_columns:
        st.warning(f"Crop_Desc.csv is missing columns: {missing_columns}. Using defaults where missing.")
        for col in missing_columns:
            if col not in ['label', 'image']:
                df_desc[col] = None
except Exception as e:
    st.error(f"Error loading Crop_Desc.csv: {str(e)}")
    df_desc = pd.DataFrame(columns=[
        'label', 'image', 'N_min', 'N_max',
        'P_min', 'P_max', 'K_min', 'K_max',
        'pH_min', 'pH_max'
    ])

# Image handling functions
def extract_image_url(html_string):
    if pd.isna(html_string) or not html_string:
        return None
    if '<img' in html_string.lower():
        match = re.search(r'src=["\'](.*?)["\']', html_string)
        return match.group(1) if match else None
    return html_string


def load_image(image_url):
    if not image_url:
        return None
    if image_url.startswith('images/') and os.path.exists(image_url):
        return image_url
    try:
        headers = {
            'User-Agent': 'DynamicCropRecommendation/1.0 (http://yourwebsite.com; contact@yourwebsite.com)'
        }
        response = requests.get(image_url, headers=headers, timeout=5)
        response.raise_for_status()
        return image_url
    except Exception as e:
        st.warning(f"Failed to load image: {str(e)}. Using placeholder.")
        return None


def get_fallback_image(crop_name):
    crop_name_clean = crop_name.lower().replace(' ', '_')
    local_path = f"images/{crop_name_clean}.jpg"
    if os.path.exists(local_path):
        return local_path
    return "https://via.placeholder.com/300x200.png?text=" + crop_name_clean.replace("_", "+")


# Weather API functions
def get_weather_data(city):
    api_key = "625ada215c472ef8b66b4261ae3d4c14"  # Replace with your OpenWeatherMap API key
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["cod"] == 200:
            rainfall = data.get("rain", {}).get("1h", 0) if data.get("rain") else "Not available"
            weather = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rainfall": rainfall if isinstance(rainfall, (int, float)) else 0,
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
            return weather
        else:
            st.warning(f"Weather data not found for {city}. API response: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None


def get_weather_forecast(city):
    api_key = "625ada215c472ef8b66b4261ae3d4c14"
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["cod"] == "200":
            forecast = []
            for item in data["list"][:5]:
                forecast.append({
                    "time": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "rainfall": item.get("rain", {}).get("3h", 0)
                })
            return forecast
        else:
            st.warning(f"Forecast not found for {city}. API response: {data.get('message', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Error fetching forecast data: {str(e)}")
        return None


# PDF report generation
def generate_pdf_report(data, top_crops, top_probs, soil_insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='Title', fontSize=16, leading=20, textColor=colors.black, alignment=1, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        name='Heading', fontSize=12, leading=16, textColor=colors.black, spaceAfter=10
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    normal_style.textColor = colors.black

    elements = []
    elements.append(Paragraph("Crop Recommendation Report", title_style))
    elements.append(Paragraph(f"Generated on: {data['Timestamp']}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Input Parameters table
    elements.append(Paragraph("Input Parameters", heading_style))
    input_data = [
        ["Parameter", "Value"],
        ["Nitrogen (N)", f"{data['N']} kg/ha"],
        ["Phosphorus (P)", f"{data['P']} kg/ha"],
        ["Potassium (K)", f"{data['K']} kg/ha"],
        ["Temperature", f"{data['Temperature']} °C"],
        ["Humidity", f"{data['Humidity']} %"],
        ["pH", f"{data['pH']}"],
        ["Rainfall", f"{data['Rainfall']} mm"],
        ["Location", data['Location']]
    ]
    input_table = Table(input_data, colWidths=[2.5 * inch, 3.5 * inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Prediction Results
    elements.append(Paragraph("Prediction Results", heading_style))
    prediction_data = [
        ["Parameter", "Value"],
        ["Recommended Crop", data['Predicted Crop']],
        ["Confidence", f"{data['Confidence (%)']:.2f}%"]
    ]
    prediction_table = Table(prediction_data, colWidths=[2.5 * inch, 3.5 * inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(prediction_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Soil Health Insights
    elements.append(Paragraph("Soil Health Insights", heading_style))
    insights_data = [["Insight"]] + [[insight] for insight in soil_insights]
    insights_table = Table(insights_data, colWidths=[6 * inch])
    insights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(insights_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Top Crops
    elements.append(Paragraph("Top Crop Recommendations", heading_style))
    top_data = [["Rank", "Crop", "Confidence"]]
    for i, (crop, prob) in enumerate(zip(top_crops, top_probs), 1):
        top_data.append([str(i), crop, f"{prob:.2f}%"])
    top_table = Table(top_data, colWidths=[1 * inch, 3 * inch, 2 * inch])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(top_table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Generated by Crop Recommendation System", normal_style))
    doc.build(elements)
    buffer.seek(0)
    return buffer


# Soil Health Insights Function
def get_soil_health_insights(n, p, k, ph, predicted_crop):
    crop_info = df_desc[df_desc['label'].str.lower() == predicted_crop.lower()]
    if not crop_info.empty:
        ranges = {
            "N": (float(crop_info['N_min'].iloc[0]), float(crop_info['N_max'].iloc[0])),
            "P": (float(crop_info['P_min'].iloc[0]), float(crop_info['P_max'].iloc[0])),
            "K": (float(crop_info['K_min'].iloc[0]), float(crop_info['K_max'].iloc[0])),
            "ph": (float(crop_info['pH_min'].iloc[0]), float(crop_info['pH_max'].iloc[0]))
        }
    else:
        ranges = {"N": (50, 100), "P": (20, 50), "K": (20, 50), "ph": (6.0, 7.0)}

    insights = []
    if "N" in ranges:
        if n < ranges["N"][0]:
            insights.append(f"Nitrogen ({n} kg/ha) is low. Consider adding urea or organic manure to reach {ranges['N'][0]}-{ranges['N'][1]} kg/ha.")
        elif n > ranges["N"][1]:
            insights.append(f"Nitrogen ({n} kg/ha) is high. Reduce fertilizer use to stay within {ranges['N'][0]}-{ranges['N'][1]} kg/ha.")
    if "P" in ranges:
        if p < ranges["P"][0]:
            insights.append(f"Phosphorus ({p} kg/ha) is low. Apply superphosphate to reach {ranges['P'][0]}-{ranges['P'][1]} kg/ha.")
        elif p > ranges["P"][1]:
            insights.append(f"Phosphorus ({p} kg/ha) is high. Avoid over-fertilization to stay within {ranges['P'][0]}-{ranges['P'][1]} kg/ha.")
    if "K" in ranges:
        if k < ranges["K"][0]:
            insights.append(f"Potassium ({k} kg/ha) is low. Use potash fertilizers to reach {ranges['K'][0]}-{ranges['K'][1]} kg/ha.")
        elif k > ranges["K"][1]:
            insights.append(f"Potassium ({k} kg/ha) is high. Reduce potassium inputs to stay within {ranges['K'][0]}-{ranges['K'][1]} kg/ha.")
    if "ph" in ranges:
        if ph < ranges["ph"][0]:
            insights.append(f"Soil pH ({ph}) is too acidic. Add lime to raise it to {ranges['ph'][0]}-{ranges['ph'][1]}.")
        elif ph > ranges["ph"][1]:
            insights.append(f"Soil pH ({ph}) is too alkaline. Add sulfur or organic matter to lower it to {ranges['ph'][0]}-{ranges['ph'][1]}.")

    if not insights:
        insights.append("Soil health is within ideal ranges for this crop based on available data!")
    return insights


# Main App Layout
st.markdown("<h1 style='text-align: center;'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Predict", "History", "Feedback"])

with tab1:
    st.markdown("<div class='card'><h3>Enter Soil and Weather Parameters</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        n_input = st.number_input(
            "Nitrogen (N) in kg/ha",
            min_value=0.0, max_value=140.0, step=0.1,
            help="Nitrogen content in soil (0-140 kg/ha)."
        )
        p_input = st.number_input(
            "Phosphorus (P) in kg/ha",
            min_value=5.0, max_value=145.0, step=0.1,
            help="Phosphorus content in soil (5-145 kg/ha)."
        )
        k_input = st.number_input(
            "Potassium (K) in kg/ha",
            min_value=5.0, max_value=205.0, step=0.1,
            help="Potassium content in soil (5-205 kg/ha)."
        )
        ph_input = st.number_input(
            "Soil pH",
            min_value=3.6, max_value=9.9, step=0.1,
            help="Soil pH level (3.6-9.9)."
        )

    with col2:
        city = st.text_input(
            "City Name",
            placeholder="Enter a city (e.g., Delhi, Mumbai)",
            help="Enter the city name to fetch real-time weather data."
        )
        location = st.selectbox(
            "Region",
            [
                'Central India', 'Eastern India', 'North Eastern India',
                'Northern India', 'Southern India', 'Western India', 'Other'
            ],
            help="Select the geographical region."
        )

    weather_data = get_weather_data(city) if city else None

    temp_input = weather_data["temperature"] if weather_data and 9 <= weather_data["temperature"] <= 43 else 9.0
    hum_input = weather_data["humidity"] if weather_data and 15 <= weather_data["humidity"] <= 99 else 15.0
    rain_input = weather_data["rainfall"] if weather_data and 21 <= weather_data["rainfall"] <= 1000 else 21.0

    inputs_valid = True
    if any(x < 0 for x in [n_input, p_input, k_input, ph_input]) or not city:
        st.error("Input values cannot be negative, and a city name is required.")
        inputs_valid = False

    numerical_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input]]
    try:
        numerical_inputs_scaled = scaler.transform(numerical_inputs)
    except Exception as e:
        st.error(f"Error scaling inputs: {str(e)}")
        inputs_valid = False

    region_columns = [
        'region_Central India', 'region_Eastern India', 'region_North Eastern India',
        'region_Northern India', 'region_Other', 'region_Southern India', 'region_Western India'
    ]
    region_encoding = [0] * 7
    region_map = {loc: idx for idx, loc in enumerate([
        'Central India', 'Eastern India', 'North Eastern India',
        'Northern India', 'Southern India', 'Western India', 'Other'
    ])}
    if location in region_map:
        region_encoding[region_map[location]] = 1

    feature_names = rdf_clf.feature_names_in_.tolist()
    predict_inputs = np.hstack([numerical_inputs_scaled, [region_encoding]])
    predict_df = pd.DataFrame(predict_inputs, columns=feature_names)

    if st.button("Recommend Crop", use_container_width=True):
        if not inputs_valid:
            st.error("Please correct the input errors before predicting.")
        else:
            with st.spinner("Analyzing data..."):
                try:
                    predicted_crop = rdf_clf.predict(predict_df)[0]
                    probabilities = rdf_clf.predict_proba(predict_df)[0]
                    confidence = probabilities[rdf_clf.classes_ == predicted_crop][0] * 100
                    top_indices = np.argsort(probabilities)[-3:][::-1]
                    top_crops = rdf_clf.classes_[top_indices]
                    top_probs = probabilities[top_indices] * 100

                    soil_insights = get_soil_health_insights(n_input, p_input, k_input, ph_input, predicted_crop)

                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'N': n_input, 'P': p_input, 'K': k_input,
                        'Temperature': temp_input, 'Humidity': hum_input,
                        'pH': ph_input, 'Rainfall': rain_input,
                        'Location': location, 'Predicted Crop': predicted_crop,
                        'Confidence': confidence
                    })

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Recommended Crop: <b>{predicted_crop}</b></h3>", unsafe_allow_html=True)
                    st.markdown(f"<h5>Confidence Score: {confidence:.2f}%</h5>", unsafe_allow_html=True)

                    if weather_data:
                        st.markdown("**Current Weather Conditions:**")
                        weather_df = pd.DataFrame({
                            "Parameter": ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Description", "Wind Speed (m/s)"],
                            "Value": [weather_data["temperature"], weather_data["humidity"], weather_data["rainfall"],
                                      weather_data["description"], weather_data["wind_speed"]]
                        })
                        st.table(weather_df)
                    else:
                        st.warning("Weather data not available. Using default values.")

                    st.markdown("**Soil Health Insights:**")
                    for insight in soil_insights:
                        st.markdown(f"- {insight}")

                    st.markdown("**Top 3 Crop Recommendations:**")
                    top_df = pd.DataFrame({
                        'Crop': top_crops,
                        'Confidence (%)': [f"{prob:.2f}" for prob in top_probs]
                    })
                    if not top_df.empty:
                        st.markdown(
                            f"""
                            <table class="custom-table">
                            <tr><th>Crop</th><th>Confidence (%)</th></tr>
                            {"".join(f"<tr><td>{row['Crop']}</td><td>{row['Confidence (%)']}</td></tr>" for _, row in top_df.iterrows())}
                            </table>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("**Top Crop Images:**")
                    cols = st.columns(3)
                    for idx, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                        with cols[idx]:
                            crop_clean = crop.lower().strip()
                            df_labels = [label.lower().strip() for label in df_desc['label'].values]
                            if crop_clean in df_labels:
                                original_label = df_desc['label'].iloc[df_labels.index(crop_clean)]
                                crop_info = df_desc[df_desc['label'] == original_label]
                                image_html = crop_info['image'].iloc[0]
                                image_url = extract_image_url(image_html)
                                valid_url = load_image(image_url) if image_url else None
                                if valid_url:
                                    st.image(valid_url, caption=f"{crop} ({prob:.2f}%)", width=200)
                                else:
                                    st.image(get_fallback_image(crop), caption=f"{crop} ({prob:.2f}%) (Placeholder)", width=200)
                            else:
                                st.warning(f"No image data found for '{crop}' in Crop_Desc.csv.")
                                st.image(get_fallback_image(crop), caption=f"{crop} ({prob:.2f}%) (Placeholder)", width=200)

                    st.markdown("**Confidence Distribution**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=top_probs, y=top_crops, palette='Greens')
                    ax.set_xlabel("Confidence (%)")
                    ax.set_title("Top Crop Probabilities")
                    st.pyplot(fig)

                    st.markdown("**Click below to download the entire report!**")
                    result_df = pd.DataFrame({
                        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'N': [n_input], 'P': [p_input], 'K': [k_input],
                        'Temperature': [temp_input], 'Humidity': [hum_input],
                        'pH': [ph_input], 'Rainfall': [rain_input],
                        'Location': [location], 'Predicted Crop': [predicted_crop],
                        'Confidence (%)': [confidence]
                    })
                    st.download_button(
                        "Download Report",
                        generate_pdf_report(result_df.to_dict('records')[0], top_crops, top_probs, soil_insights),
                        f"crop_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'><h3>Prediction History</h3>", unsafe_allow_html=True)
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'><h3>User Feedback</h3>", unsafe_allow_html=True)
    st.markdown("We value your feedback to improve the system!")

    with st.form("feedback_form"):
        feedback_text = st.text_area("Your Feedback", placeholder="Share your thoughts or suggestions...")
        feedback_rating = st.slider("Rate your experience", 1, 5, 3, help="1 = Poor, 5 = Excellent")
        submit_feedback = st.form_submit_button("Submit Feedback")
        if submit_feedback:
            feedback_data = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Rating': feedback_rating,
                'Feedback': feedback_text
            }
            feedback_df = pd.DataFrame([feedback_data])
            feedback_file = "feedback.csv"
            if os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, index=False)
            st.markdown(
                '<div style="background-color: #d4edda; color: #37474f; padding: 10px; border-radius: 5px; border: 1px solid #c3e6cb;">'
                'Thank you for your feedback! It has been recorded.'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("**Feedback Viewer (Admin Only)**")
    admin_password = st.text_input("Enter Admin Password", type="password", key="admin_password")
    if admin_password:
        if admin_password == "admin123":
            st.session_state.admin_authenticated = True
        else:
            st.error("Incorrect password.")

    if st.session_state.admin_authenticated:
        feedback_file = "feedback.csv"
        if os.path.exists(feedback_file):
            feedback_df = pd.read_csv(feedback_file)
            st.markdown("### All Feedback Entries")
            st.dataframe(feedback_df, use_container_width=True)
        else:
            st.info("No feedback submitted yet.")

        if os.path.exists(feedback_file):
            with open(feedback_file, "rb") as f:
                st.download_button(
                    "Download Feedback Summary",
                    f,
                    file_name="feedback_summary.csv",
                    mime="text/csv",
                    help="Download all feedback entries as a CSV file."
                )
    else:
        st.info("Please enter the admin password to view feedback.")
    st.markdown("</div>", unsafe_allow_html=True)

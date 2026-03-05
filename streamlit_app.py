import streamlit as st
import sqlite3
from datetime import datetime
import os
import streamlit.components.v1 as components
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import math

# QR CODE IMPORTS
try:
    import qrcode
    from io import BytesIO
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# ================================================================
#  CONFIGURATION & SETUP
# ================================================================

DB_FILE = "aurora_reports.db"
IMAGE_FOLDER = "evidence_images"
MODEL_FILE = "priority_model.pkl"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

st.set_page_config(page_title="Aurora Sentinel", page_icon="🚨", layout="wide")

# ================================================================
#  CSS STYLES
# ================================================================

st.markdown("""
<style>
@keyframes flash {
    0% {background-color: #ff4d4d;}
    50% {background-color: white;}
    100% {background-color: #ff4d4d;}
}

.flash-box {
    animation: flash 1s infinite;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
    color: black;
    border: 2px solid #cc0000;
}

.alert-badge {
    background-color: #ff4d4d;
    color: white;
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Receipt Style */
.receipt-box {
    border: 2px dashed #555;
    padding: 25px;
    border-radius: 10px;
    background-color: #f9f9f9;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.receipt-id {
    font-size: 32px;
    font-weight: 900;
    color: #d32f2f;
    letter-spacing: 2px;
    background-color: #fff;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    display: inline-block;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
#  DATABASE SETUP
# ================================================================

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT, crime_type TEXT, latitude REAL, longitude REAL,
    place_name TEXT, location TEXT, duration TEXT, description TEXT,
    evidence TEXT, priority TEXT, status TEXT, timestamp TEXT,
    assigned_station TEXT
)""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS sos_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT, latitude REAL, longitude REAL,
    place_name TEXT, status TEXT, timestamp TEXT
)""")
conn.commit()

station_passkeys = {
    "Hyderabad":        "HYD2024", "Hitech City":      "HITECH2024",
    "Gachibowli":       "GACH2024", "Miyapur":          "MIYA2024",
    "Kukatpally":       "KUKA2024", "Secunderabad":     "SEC2024",
    "Begumpet":         "BEGU2024", "LB Nagar":         "LBN2024",
    "Banjara Hills":    "BANJ2024", "Jubilee Hills":    "JUB2024",
    "Ameerpet":         "AMEE2024", "Somajiguda":       "SOMA2024"
}

AREA_TO_STATION = {k: k for k in station_passkeys.keys()}

# ================================================================
#  SMART ROUTING: STATION COORDINATES
# ================================================================

STATION_COORDS = {
    "Hyderabad": (17.3850, 78.4867),
    "Hitech City": (17.4615, 78.3775),
    "Gachibowli": (17.4400, 78.3480),
    "Miyapur": (17.4969, 78.3683),
    "Kukatpally": (17.4849, 78.4074),
    "Secunderabad": (17.4399, 78.4983),
    "Begumpet": (17.4435, 78.4667),
    "LB Nagar": (17.3514, 78.5533),
    "Banjara Hills": (17.4156, 78.4347),
    "Jubilee Hills": (17.4325, 78.4073),
    "Ameerpet": (17.4375, 78.4483),
    "Somajiguda": (17.4230, 78.4560)
}

def get_nearest_station(lat, lon):
    min_dist = float('inf')
    nearest = "Hyderabad"
    for station, coords in STATION_COORDS.items():
        s_lat, s_lon = coords
        dist = math.sqrt((lat - s_lat)**2 + (lon - s_lon)**2)
        if dist < min_dist:
            min_dist = dist
            nearest = station
    return nearest

# ================================================================
#  INTELLIGENT MACHINE LEARNING MODULE
# ================================================================

def train_and_save_model():
    data = [
        ('Theft', 'Chain snatching by two men on bike with knife', 'High'),
        ('Robbery', 'Armed robbery at gunpoint in jewelry shop', 'High'),
        ('Assault', 'Group of men attacked him with rods bleeding heavily', 'High'),
        ('Theft', 'Pickpocket threatened to stab if resisted', 'High'),
        ('Kidnapping', 'Child taken by unknown persons in a white van', 'High'),
        ('Kidnapping', 'Woman kidnapped near bus stop', 'High'),
        ('Murder', 'Body found with stab wounds', 'High'),
        ('Cyber Crime', 'Large scale banking fraud affecting thousands', 'High'),
        ('Harassment', 'Stalking victim threatened with death', 'High'),
        ('Domestic Violence', 'Husband attacked wife with hot oil severe burns', 'High'),
        ('Theft', 'Stole a laptop from an unattended bag', 'Medium'),
        ('Assault', 'Minor scuffle during a traffic argument', 'Medium'),
        ('Fraud', 'Online shopping product not delivered fake site', 'Medium'),
        ('Harassment', 'Neighbor playing loud music daily causing disturbance', 'Medium'),
        ('Theft', 'Bicycle stolen from apartment basement', 'Medium'),
        ('Theft', 'Petty theft of vegetables from cart', 'Low'),
        ('Theft', 'Mobile phone lost in park probably fell', 'Low'),
        ('Other', 'Noise complaint regarding late night party', 'Low'),
        ('Other', 'Stray dog menace in the street', 'Low'),
        ('Fraud', 'Small amount overcharged in retail shop', 'Low'),
        ('Theft', 'Lost wallet returned by neighbor no cash missing', 'Low'),
        ('Theft', 'Shoplifting incident involved group creating chaos', 'Medium'),
        ('Assault', 'Verbal abuse and slap during road rage', 'Medium'),
        ('Murder', 'Suspected murder case found in field', 'High'),
        ('Emergency', 'One tap emergency image submission', 'High'),
    ]
    
    df = pd.DataFrame(data, columns=['crime_type', 'description', 'priority'])
    X = df[['crime_type', 'description']]
    y = df['priority']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['crime_type']),
            ('text', TfidfVectorizer(stop_words='english', ngram_range=(1,2)), 'description')
        ], remainder='passthrough')
    model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

try:
    ml_model = joblib.load(MODEL_FILE)
except:
    ml_model = train_and_save_model()

def get_priority_ml(crime_type, description):
    critical_crimes = ["Kidnapping", "Murder", "Emergency", "Robbery", "Domestic Violence"]
    if crime_type in critical_crimes:
        return "High"
    try:
        if not description or description.strip() == "": desc = "general incident"
        else: desc = description
        input_data = pd.DataFrame({'crime_type': [crime_type], 'description': [desc]})
        prediction = ml_model.predict(input_data)
        return prediction[0]
    except:
        high = ["Assault","Cyber Crime","Harassment","Robbery","Kidnapping","Domestic Violence","Murder"]
        if crime_type in high: return "High"
        return "Medium"

# ================================================================
#  HELPER: SMART ADDRESS FETCHER
# ================================================================

def is_coords_only(text):
    if not text: return True
    return not any(c.isalpha() for c in text)

def get_place_name(lat, lon):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", 
                         params={"lat": lat, "lon": lon, "format": "json"}, 
                         headers={"User-Agent": "AuroraSentinel/1.0"}, 
                         timeout=10)
        data = r.json()
        return data.get("display_name", f"{lat:.5f}, {lon:.5f}")
    except:
        return f"{lat:.5f}, {lon:.5f}"

# ================================================================
#  TRANSLATIONS
# ================================================================

ALL_T = {
    "English": {
        "title":            "🚨 Aurora Sentinel – Crime Reporting & Tracking System",
        "nav_label":        "📂 Navigation",
        "navigation":       ["Report Crime", "Track Case", "Admin Panel", "SOS"],
        "report_crime":     "📢 Report a Crime",
        "crime_type":       "Select Crime Type",
        "location":         "Location",
        "duration":         "Incident Duration",
        "incident":         "Incident Occurred At",
        "description":      "Description (Optional)",
        "evidence_section": "📎 Evidence",
        "evidence_type":    "Select Evidence Type",
        "image":            "Image",
        "pdf":              "PDF",
        "upload_images":    "Upload Evidence Images",
        "upload_pdf":       "Upload PDF Evidence",
        "submit":           "Submit Report",
        "live_location_info": "📍 Click below to auto-detect & fill your GPS coordinates",
        "lat_input":        "Latitude",
        "lon_input":        "Longitude",
        "lat_placeholder":  "Auto-filled by GPS button above",
        "lon_placeholder":  "Auto-filled by GPS button above",
        "report_success":   "✅ Report submitted successfully",
        "location_error":   "❌ Invalid coordinates. Please enter valid numbers.",
        "detected_address": "📍 Detected Address:",
        "track_title":      "🔍 Track Case",
        "select_case":      "📂 Select a Reported Case",
        "track_btn":        "Track",
        "case_found":       "✅ Case Found",
        "case_not_found":   "❌ Case not found",
        "case_id_label":    "📌 Case ID:",
        "location_label":   "📍 Location:",
        "gps_place":        "📍 GPS Address:",
        "duration_label":   "⏱️ Duration:",
        "lat_label":        "🌐 Latitude:",
        "lon_label":        "🌐 Longitude:",
        "status_label":     "📊 Status:",
        "priority_label":   "🔥 Priority:",
        "time_label":       "🕒 Time:",
        "evidence_images":  "🖼 Evidence Images:",
        "image_not_found":  "⚠️ Image file not found",
        "evidence_file":    "📄 Evidence File:",
        "no_evidence":      "📂 No evidence uploaded",
        "no_cases":         "No cases reported yet.",
        "no_gps":           "📍 GPS location not recorded for this case.",
        "admin_title":      "🔐 Admin Panel",
        "station":          "Select Police Station",
        "passkey":          "Enter Passkey",
        "login":            "Login",
        "login_success":    "✅ Login successful",
        "login_fail":       "❌ Invalid credentials",
        "logged_in_as":     "Logged in as",
        "dashboard":        "📊 Dashboard",
        "crime_reports_tab":"📋 Crime Reports",
        "sos_tab":          "🚨 SOS Alerts",
        "total_cases":      "Total Cases",
        "urgent_cases":     "High Priority Cases",
        "closed_cases":     "Closed Cases",
        "no_reports":       "No crime reports found.",
        "description_lbl":  "📝 Description:",
        "no_description":   "No description",
        "evidence_lbl":     "🖼 Evidence:",
        "no_evidence_up":   "No evidence uploaded",
        "update_status":    "Update Status",
        "save":             "💾 Save",
        "status_updated":   "✅ Status updated",
        "no_sos":           "No SOS alerts.",
        "logout":           "🚪 Logout",
        "chart_main":       "📊 System Overview: Cases, Emergencies & SOS",
        "no_data_chart":    "No data available for charts yet.",
        "gps_loc_label":    "📍 GPS Location:",
        "gps_coords":       "🌐 GPS Coordinates:",
        "sos_title":        "🚨 SOS Emergency",
        "sos_gps_info":     "📍 Click the button to auto-detect GPS and fill coordinates automatically.",
        "sos_coords_title": "#### 📋 GPS Coordinates (auto-filled above)",
        "sos_lat_input":    "SOS Latitude",
        "sos_lon_input":    "SOS Longitude",
        "sos_lat_ph":       "Auto-filled by button above",
        "sos_lon_ph":       "Auto-filled by button above",
        "send_sos":         "🚨 Send SOS",
        "sos_sent":         "🚨 SOS sent successfully!",
        "sos_no_coords":    "❌ Please detect GPS first – coordinates are required.",
        "sos_your_loc":     "📍 Your Location:",
        "gps_btn_report":   "📍 Auto-Detect My GPS Location",
        "gps_btn_sos":      "🚨 Detect My GPS Location & Auto-Fill",
        "status_options":   ["Under Review", "Investigating", "Action Taken", "Closed"],
        "crime_types":      ["Harassment","Theft","Robbery","Assault","Cyber Crime","Fraud",
                             "Kidnapping","Domestic Violence","Murder","Other"],
        "locations":        ["Hyderabad","Hitech City","Gachibowli","Miyapur","Kukatpally",
                             "Secunderabad","Begumpet","LB Nagar","Banjara Hills",
                             "Jubilee Hills","Ameerpet","Somajiguda"],
        "durations":        ["Just Happened","Less than 15 minutes","15-30 minutes",
                             "30 minutes - 1 hour","1-2 hours","2-4 hours","More than 4 hours",
                             "Half Day","Full Day","Multiple Days","Ongoing"],
        "incidents":        ["Home","School","Workplace","Street","Other"],
        "one_tap_title":    "🚨 One-Tap Emergency Report",
        "upload_emg_img":   "Upload Emergency Image",
        "emg_submitted":    "🚨 Emergency Submitted!",
        "routed_to":        "🚔 Routed to",
        "ml_info":          "🤖 AI Priority Prediction Active (Learns from Description)",
        "legend_high_crimes": "High Priority Crimes",
        "legend_emergencies": "Emergencies",
        "legend_sos": "SOS Alerts",
        "legend_medium": "Medium Priority",
        "legend_low": "Low Priority",
        "html_loc_detected": "Location Detected!",
        "html_lat": "Lat",
        "html_lon": "Lon",
        "html_err_denied": "❌ Location access denied.",
        "html_detecting": "📡 Detecting…",
        "html_btn_detected": "✅ GPS Detected",
        "fetching_addr": "📍 Fetching GPS address...",
        "map_title": "📍 Live Case Locations Map",
        "no_gps_map": "No GPS data available to display on map.",
        "critical_alert": "🚨 CRITICAL CASES DETECTED - IMMEDIATE ACTION REQUIRED 🚨",
        "sidebar_alert": "🚨 {} High Priority Cases",
        # Receipt System - UPDATED TEXT
        "receipt_title": "🧾 CRIME REPORT RECEIPT",
        "receipt_warning": "⚠️ IMPORTANT: Save this Report ID by taking a screenshot in your device to track your case status.",
        "track_id_prompt": "Enter your Report ID (Case ID):",
        # QR Code
        "qr_title": "📡 Share App",
        "qr_caption": "Scan to open on device",
        # Warning text for GPS
        "html_warn_enter": "⚠️ IMPORTANT: Click boxes below & press ENTER.",
        # Close button
        "close_case_btn": "🔒 Close Case"
    },
    "Hindi": {
        "title":            "🚨 ऑरोरा सेंटिनल – अपराध रिपोर्टिंग और ट्रैकिंग सिस्टम",
        "nav_label":        "📂 नेविगेशन",
        "navigation":       ["अपराध रिपोर्ट करें", "केस ट्रैक करें", "एडमिन पैनल", "SOS"],
        "report_crime":     "📢 अपराध की रिपोर्ट करें",
        "crime_type":       "अपराध का प्रकार चुनें",
        "location":         "स्थान",
        "duration":         "घटना की अवधि",
        "incident":         "घटना कहाँ हुई",
        "description":      "विवरण (वैकल्पिक)",
        "evidence_section": "📎 साक्ष्य",
        "evidence_type":    "साक्ष्य प्रकार चुनें",
        "image":            "चित्र",
        "pdf":              "PDF",
        "upload_images":    "साक्ष्य चित्र अपलोड करें",
        "upload_pdf":       "PDF साक्ष्य अपलोड करें",
        "submit":           "रिपोर्ट सबमिट करें",
        "live_location_info": "📍 GPS स्थान स्वतः पता करने के लिए नीचे क्लिक करें",
        "lat_input":        "अक्षांश",
        "lon_input":        "देशांतर",
        "lat_placeholder":  "GPS बटन से स्वतः भरा जाएगा",
        "lon_placeholder":  "GPS बटन से स्वतः भरा जाएगा",
        "report_success":   "✅ रिपोर्ट सफलतापूर्वक सबमिट की गई",
        "location_error":   "❌ अमान्य निर्देशांक। कृपया सही नंबर दर्ज करें।",
        "detected_address": "📍 पता पता चला:",
        "track_title":      "🔍 केस ट्रैक करें",
        "select_case":      "📂 रिपोर्ट किया गया केस चुनें",
        "track_btn":        "ट्रैक करें",
        "case_found":       "✅ केस मिला",
        "case_not_found":   "❌ केस नहीं मिला",
        "case_id_label":    "📌 केस ID:",
        "location_label":   "📍 स्थान:",
        "gps_place":        "📍 GPS पता:",
        "duration_label":   "⏱️ अवधि:",
        "lat_label":        "🌐 अक्षांश:",
        "lon_label":        "🌐 देशांतर:",
        "status_label":     "📊 स्थिति:",
        "priority_label":   "🔥 प्राथमिकता:",
        "time_label":       "🕒 समय:",
        "evidence_images":  "🖼 साक्ष्य चित्र:",
        "image_not_found":  "⚠️ चित्र फ़ाइल नहीं मिली",
        "evidence_file":    "📄 साक्ष्य फ़ाइल:",
        "no_evidence":      "📂 कोई साक्ष्य अपलोड नहीं",
        "no_cases":         "अभी तक कोई केस दर्ज नहीं।",
        "no_gps":           "📍 इस केस के लिए GPS स्थान दर्ज नहीं है।",
        "admin_title":      "🔐 एडमिन पैनल",
        "station":          "पुलिस स्टेशन चुनें",
        "passkey":          "पासकी दर्ज करें",
        "login":            "लॉगिन",
        "login_success":    "✅ लॉगिन सफल",
        "login_fail":       "❌ अमान्य क्रेडेंशियल",
        "logged_in_as":     "लॉग इन किया गया",
        "dashboard":        "📊 डैशबोर्ड",
        "crime_reports_tab":"📋 अपराध रिपोर्ट",
        "sos_tab":          "🚨 SOS अलर्ट",
        "total_cases":      "कुल केस",
        "urgent_cases":     "उच्च प्राथमिकता केस",
        "closed_cases":     "बंद केस",
        "no_reports":       "कोई अपराध रिपोर्ट नहीं।",
        "description_lbl":  "📝 विवरण:",
        "no_description":   "कोई विवरण नहीं",
        "evidence_lbl":     "🖼 साक्ष्य:",
        "no_evidence_up":   "कोई साक्ष्य अपलोड नहीं",
        "update_status":    "स्थिति अपडेट करें",
        "save":             "💾 सहेजें",
        "status_updated":   "✅ स्थिति अपडेट हुई",
        "no_sos":           "कोई SOS अलर्ट नहीं।",
        "logout":           "🚪 लॉगआउट",
        "chart_main":       "📊 सिस्टम अवलोकन: केस, इमरजेंसी और SOS",
        "no_data_chart":    "चार्ट के लिए अभी तक डेटा उपलब्ध नहीं।",
        "gps_loc_label":    "📍 GPS स्थान:",
        "gps_coords":       "🌐 GPS निर्देशांक:",
        "sos_title":        "🚨 SOS आपातकाल",
        "sos_gps_info":     "📍 GPS स्थान स्वतः पता करने के लिए बटन दबाएं।",
        "sos_coords_title": "#### 📋 GPS निर्देशांक (ऊपर से स्वतः भरे जाएंगे)",
        "sos_lat_input":    "SOS अक्षांश",
        "sos_lon_input":    "SOS देशांतर",
        "sos_lat_ph":       "बटन से स्वतः भरा जाएगा",
        "sos_lon_ph":       "बटन से स्वतः भरा जाएगा",
        "send_sos":         "🚨 SOS भेजें",
        "sos_sent":         "🚨 SOS सफलतापूर्वक भेजा गया!",
        "sos_no_coords":    "❌ पहले GPS से स्थान पता करें।",
        "sos_your_loc":     "📍 आपका स्थान:",
        "gps_btn_report":   "📍 मेरा GPS स्थान स्वतः पता करें",
        "gps_btn_sos":      "🚨 GPS स्थान पता करें और भरें",
        "status_options":   ["समीक्षाधीन","जांच में","कार्रवाई की गई","बंद"],
        "crime_types":      ["उत्पीड़न","चोरी","डकैती","हमला","साइबर अपराध","धोखाधड़ी",
                             "अपहरण","घरेलू हिंसा","हत्या","अन्य"],
        "locations":        ["हैदराबाद","हाईटेक सिटी","गाचीबौली","मियापुर","कुकटपल्ली",
                             "सिकंदराबाद","बेगमपेट","एलबी नगर","बंजारा हिल्स",
                             "जुबली हिल्स","अमीरपेट","सोमाजिगुड़ा"],
        "durations":        ["अभी हुआ","15 मिनट से कम","15-30 मिनट","30 मिनट - 1 घंटा",
                             "1-2 घंटे","2-4 घंटे","4 घंटे से अधिक","आधा दिन",
                             "पूरा दिन","कई दिन","जारी है"],
        "incidents":        ["घर","स्कूल","कार्यस्थल","सड़क","अन्य"],
        "one_tap_title":    "🚨 वन-टैप इमरजेंसी रिपोर्ट",
        "upload_emg_img":   "इमरजेंसी फोटो अपलोड करें",
        "emg_submitted":    "🚨 इमरजेंसी रिपोर्ट दर्ज!",
        "routed_to":        "🚔 भेजा गया:",
        "ml_info":          "🤖 AI प्राथमिकता भविष्यवाणी सक्रिय (विवरण से सीखता है)",
        "legend_high_crimes": "उच्च प्राथमिकता अपराध",
        "legend_emergencies": "आपातकालीन रिपोर्ट",
        "legend_sos": "SOS अलर्ट",
        "legend_medium": "मध्यम प्राथमिकता",
        "legend_low": "निम्न प्राथमिकता",
        "html_loc_detected": "स्थान पता चला!",
        "html_lat": "अक्षांश",
        "html_lon": "देशांतर",
        "html_err_denied": "❌ स्थान अनुमति अस्वीकार।",
        "html_detecting": "📡 पता लगा रहा है…",
        "html_btn_detected": "✅ GPS पता चला",
        "fetching_addr": "📍 GPS पता प्राप्त कर रहे हैं...",
        "map_title": "📍 लाइव केस लोकेशन मैप",
        "no_gps_map": "मैप पर दिखाने के लिए कोई GPS डेटा उपलब्ध नहीं।",
        "critical_alert": "🚨 गंभीर मामले पाए गए - तत्काल कार्रवाई आवश्यक 🚨",
        "sidebar_alert": "🚨 {} उच्च प्राथमिकता केस",
        # Receipt System
        "receipt_title": "🧾 अपराध रिपोर्ट रसीद",
        "receipt_warning": "⚠️ महत्वपूर्ण: अपनी रिपोर्ट की स्थिति ट्रैक करने के लिए स्क्रीनशॉट लेकर इस रिपोर्ट ID को सेव करें।",
        "track_id_prompt": "अपनी रिपोर्ट ID (केस ID) दर्ज करें:",
        # QR Code
        "qr_title": "📡 ऐप शेयर करें",
        "qr_caption": "डिवाइस पर खोलने के लिए स्कैन करें",
        # Warning text for GPS
        "html_warn_enter": "⚠️ महत्वपूर्वपूर्ण: नीचे दिए बॉक्स पर क्लिक करें और ENTER दबाएं।",
        # Close button
        "close_case_btn": "🔒 केस बंद करें"
    },
    "Telugu": {
        "title":            "🚨 ఆరోరా సెంటినల్ – నేర నివేదన & ట్రాకింగ్ వ్యవస్థ",
        "nav_label":        "📂 నావిగేషన్",
        "navigation":       ["నేరం నివేదించు", "కేస్ ట్రాక్ చేయి", "అడ్మిన్ పానెల్", "SOS"],
        "report_crime":     "📢 నేరం నివేదించు",
        "crime_type":       "నేరం రకం ఎంచుకోండి",
        "location":         "స్థలం",
        "duration":         "సంఘటన వ్యవధి",
        "incident":         "సంఘటన జరిగిన స్థలం",
        "description":      "వివరణ (ఐచ్ఛికం)",
        "evidence_section": "📎 సాక్ష్యాలు",
        "evidence_type":    "సాక్ష్య రకం ఎంచుకోండి",
        "image":            "చిత్రం",
        "pdf":              "PDF",
        "upload_images":    "సాక్ష్య చిత్రాలు అప్‌లోడ్ చేయండి",
        "upload_pdf":       "PDF సాక్ష్యం అప్‌లోడ్ చేయండి",
        "submit":           "నివేదన సమర్పించు",
        "live_location_info": "📍 GPS స్థానాన్ని స్వయంచాలకంగా గుర్తించడానికి దిగువ నొక్కండి",
        "lat_input":        "అక్షాంశం",
        "lon_input":        "రేఖాంశం",
        "lat_placeholder":  "GPS బటన్ ద్వారా స్వయంచాలకంగా నింపబడుతుంది",
        "lon_placeholder":  "GPS బటన్ ద్వారా స్వయంచాలకంగా నింపబడుతుంది",
        "report_success":   "✅ నివేదన విజయవంతంగా సమర్పించబడింది",
        "location_error":   "❌ చెల్లని అక్షాంశాలు. దయచేసి సరైన సంఖ్యలు నమోదు చేయండి.",
        "detected_address": "📍 గుర్తించిన చిరునామా:",
        "track_title":      "🔍 కేస్ ట్రాక్ చేయి",
        "select_case":      "📂 నివేదించిన కేస్ ఎంచుకోండి",
        "track_btn":        "ట్రాక్ చేయి",
        "case_found":       "✅ కేస్ కనుగొనబడింది",
        "case_not_found":   "❌ కేస్ కనుగొనబడలేదు",
        "case_id_label":    "📌 కేస్ ID:",
        "location_label":   "📍 స్థలం:",
        "gps_place":        "📍 GPS చిరునామా:",
        "duration_label":   "⏱️ వ్యవధి:",
        "lat_label":        "🌐 అక్షాంశం:",
        "lon_label":        "🌐 రేఖాంశం:",
        "status_label":     "📊 స్థితి:",
        "priority_label":   "🔥 ప్రాధాన్యత:",
        "time_label":       "🕒 సమయం:",
        "evidence_images":  "🖼 సాక్ష్య చిత్రాలు:",
        "image_not_found":  "⚠️ చిత్రం ఫైల్ కనుగొనబడలేదు",
        "evidence_file":    "📄 సాక్ష్య ఫైల్:",
        "no_evidence":      "📂 సాక్ష్యాలు అప్‌లోడ్ చేయబడలేదు",
        "no_cases":         "ఇంకా కేసులు నివేదించబడలేదు.",
        "no_gps":           "📍 ఈ కేస్‌కు GPS స్థానం నమోదు కాలేదు.",
        "admin_title":      "🔐 అడ్మిన్ పానెల్",
        "station":          "పోలీస్ స్టేషన్ ఎంచుకోండి",
        "passkey":          "పాస్‌కీ నమోదు చేయండి",
        "login":            "లాగిన్",
        "login_success":    "✅ లాగిన్ విజయవంతమైంది",
        "login_fail":       "❌ చెల్లని ఆధారాలు",
        "logged_in_as":     "లాగిన్ అయ్యారు",
        "dashboard":        "📊 డ్యాష్‌బోర్డ్",
        "crime_reports_tab":"📋 నేర నివేదనలు",
        "sos_tab":          "🚨 SOS హెచ్చరికలు",
        "total_cases":      "మొత్తం కేసులు",
        "urgent_cases":     "అత్యధిక ప్రాధాన్యత కేసులు",
        "closed_cases":     "మూసివేసిన కేసులు",
        "no_reports":       "నేర నివేదనలు కనుగొనబడలేదు.",
        "description_lbl":  "📝 వివరణ:",
        "no_description":   "వివరణ లేదు",
        "evidence_lbl":     "🖼 సాక్ష్యాలు:",
        "no_evidence_up":   "సాక్ష్యాలు అప్‌లోడ్ చేయబడలేదు",
        "update_status":    "స్థితి నవీకరించు",
        "save":             "💾 సేవ్ చేయి",
        "status_updated":   "✅ స్థితి నవీకరించబడింది",
        "no_sos":           "SOS హెచ్చరికలు లేవు.",
        "logout":           "🚪 లాగ్అవుట్",
        "chart_main":       "📊 సిస్టమ్ అవలోకనం: కేసులు, ఎమర్జెన్సీలు & SOS",
        "no_data_chart":    "చార్ట్‌లకు ఇంకా డేటా అందుబాటులో లేదు.",
        "gps_loc_label":    "📍 GPS స్థానం:",
        "gps_coords":       "🌐 GPS అక్షాంశాలు:",
        "sos_title":        "🚨 SOS అత్యవసర పరిస్థితి",
        "sos_gps_info":     "📍 GPS స్థానాన్ని స్వయంచాలకంగా గుర్తించడానికి బటన్ నొక్కండి.",
        "sos_coords_title": "#### 📋 GPS అక్షాంశాలు (పైన స్వయంచాలకంగా నింపబడతాయి)",
        "sos_lat_input":    "SOS అక్షాంశం",
        "sos_lon_input":    "SOS రేఖాంశం",
        "sos_lat_ph":       "బటన్ ద్వారా స్వయంచాలకంగా నింపబడుతుంది",
        "sos_lon_ph":       "బటన్ ద్వారా స్వయంచాలకంగా నింపబడుతుంది",
        "send_sos":         "🚨 SOS పంపించు",
        "sos_sent":         "🚨 SOS విజయవంతంగా పంపించబడింది!",
        "sos_no_coords":    "❌ ముందు GPS స్థానం గుర్తించండి.",
        "sos_your_loc":     "📍 మీ స్థానం:",
        "gps_btn_report":   "📍 నా GPS స్థానాన్ని స్వయంచాలకంగా గుర్తించు",
        "gps_btn_sos":      "🚨 నా GPS స్థానం గుర్తించి నింపు",
        "status_options":   ["సమీక్షలో","దర్యాప్తులో","చర్య తీసుకోబడింది","మూసివేయబడింది"],
        "crime_types":      ["హారాస్మెంట్","దొంగతనం","దోపిడీ","దాడి","సైబర్ క్రైమ్",
                             "మోసం","అపహరణ","గృహ హింస","హత్య","ఇతర"],
        "locations":        ["హైదరాబాద్","హైటెక్ సిటీ","గచ్చిబౌలి","మియాపూర్","కూకట్‌పల్లి",
                             "సికింద్రాబాద్","బేగంపేట్","ఎల్బీ నగర్","బంజారా హిల్స్",
                             "జూబ్లీ హిల్స్","అమీర్‌పేట్","సోమాజిగూడా"],
        "durations":        ["ఇప్పుడే జరిగింది","15 నిమిషాల కంటే తక్కువ","15-30 నిమిషాలు",
                             "30 నిమిషాలు - 1 గంట","1-2 గంటలు","2-4 గంటలు","4 గంటలకు పైగా",
                             "అర రోజు","పూర్తి రోజు","అనేక రోజులు","కొనసాగుతోంది"],
        "incidents":        ["ఇల్లు","పాఠశాల","కార్యాలయం","వీధి","ఇతర"],
        "one_tap_title":    "🚨 వన్-ట్యాప్ ఎమర్జెన్సీ నివేదిక",
        "upload_emg_img":   "ఎమర్జెన్సీ ఫోటో అప్‌లోడ్ చేయండి",
        "emg_submitted":    "🚨 ఎమర్జెన్సీ నమోదు చేయబడింది!",
        "routed_to":        "🚔 పంపబడింది:",
        "ml_info":          "🤖 AI ప్రాధాన్యత అంచనా క్రియాశీలంగా ఉంది (వివరణ నుండి నేర్చుకుంటుంది)",
        "legend_high_crimes": "అధిక ప్రాధాన్యత నేరాలు",
        "legend_emergencies": "అత్యవసర నివేదికలు",
        "legend_sos": "SOS హెచ్చరికలు",
        "legend_medium": "మధ్యస్థ ప్రాధాన్యత",
        "legend_low": "తక్కువ ప్రాధాన్యత",
        "html_loc_detected": "స్థానం గుర్తించబడింది!",
        "html_lat": "అక్షాంశం",
        "html_lon": "రేఖాంశం",
        "html_err_denied": "❌ స్థానం అనుమతి తిరస్కరించబడింది.",
        "html_detecting": "📡 గుర్తిస్తోంది…",
        "html_btn_detected": "✅ GPS గుర్తించబడింది",
        "fetching_addr": "📍 GPS చిరునామా తెస్తోంది...",
        "map_title": "📍 లైవ్ కేస్ లొకేషన్లు మ్యాప్",
        "no_gps_map": "మ్యాప్‌లో చూపడానికి GPS డేటా లేదు.",
        "critical_alert": "🚨 క్లిష్టమైన కేసులు గుర్తించబడ్డాయి - తక్షణ చర్య అవసరం 🚨",
        "sidebar_alert": "🚨 {} అధిక ప్రాధాన్యత కేసులు",
        # Receipt System
        "receipt_title": "🧾 నేర నివేదిక రసీదు",
        "receipt_warning": "⚠️ ముఖ్యమైనది: మీ కేసు స్థితిని ట్రాక్ చేయడానికి స్క్రీన్‌షాట్ తీసుకోవడం ద్వారా ఈ రిపోర్ట్ IDని సేవ్ చేయండి.",
        "track_id_prompt": "మీ రిపోర్ట్ ID (కేస్ ID) నమోదు చేయండి:",
        # QR Code
        "qr_title": "📡 యాప్ షేర్ చేయండి",
        "qr_caption": "పరికరంలో తెరవడానికి స్కాన్ చేయండి",
        # Warning text for GPS
        "html_warn_enter": "⚠️ ముఖ్యం: కింద ఉన్న పెట్టెలపై క్లిక్ చేసి ENTER నొక్కండి.",
        # Close button
        "close_case_btn": "🔒 కేస్ మూసివేయి"
    }
}

# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def generate_case_id(location_en):
    code = location_en[:3].upper()
    yr   = datetime.now().strftime("%y")
    cursor.execute("SELECT COUNT(*) FROM reports WHERE location LIKE ?", (f"%{location_en}%",))
    count = cursor.fetchone()[0] + 1
    return f"{code}-{yr}-{str(count).zfill(3)}"

# RESTORED YELLOW WARNING LINE
def auto_gps_component(lat_label: str, lon_label: str, btn_label: str, t_dict):
    html = f"""
    <style>
      #gps-btn{{padding:10px 22px;background:#1a73e8;color:#fff;border:none;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer;width:100%;margin-bottom:10px;}}
      #gps-btn:disabled{{background:#888;cursor:not-allowed;}}
      #coord-box{{display:none;padding:10px 14px;background:#0f2e1e;color:#00ff88;font-family:monospace;border-radius:8px;border:1px solid #00ff88;line-height:1.6;font-size:14px;}}
      #err-box{{display:none;color:#ff4444;font-size:13px;margin-top:8px;}}
      .warn-txt{{color:#ffcc00; font-weight:bold; font-size:13px; margin-top:8px; text-align:center;}}
    </style>
    <button id="gps-btn" onclick="detectGPS()">{btn_label}</button>
    <div id="coord-box">✅ <b>{t_dict['html_loc_detected']}</b><br>📍 {t_dict['html_lat']}: <span id="lat-disp">—</span> &nbsp;|&nbsp; {t_dict['html_lon']}: <span id="lon-disp">—</span>
      <div class="warn-txt">{t_dict['html_warn_enter']}</div>
    </div>
    <div id="err-box">{t_dict['html_err_denied']}</div>
    <script>
    function fillStreamlitInput(labelText, value){{
      var allInputs = window.parent.document.querySelectorAll('input[type="text"]');
      for(var i=0;i<allInputs.length;i++){{
        var inp = allInputs[i]; var container = inp.closest('[data-testid="stTextInput"]');
        if(!container) continue; var labelNode = container.querySelector('label');
        if(!labelNode) continue; var text = labelNode.innerText || labelNode.textContent;
        if(text && text.includes(labelText)){{
          inp.focus(); var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
          nativeSetter.call(inp, value); inp.dispatchEvent(new Event('input',{{bubbles:true}}));
          inp.dispatchEvent(new Event('change',{{bubbles:true}}));
          inp.dispatchEvent(new KeyboardEvent('keydown', {{key:'Enter', code:'Enter', keyCode:13, bubbles:true}}));
          inp.blur(); return;
        }}
      }}
    }}
    function detectGPS(){{
      var btn=document.getElementById('gps-btn'); var box=document.getElementById('coord-box');
      var err=document.getElementById('err-box'); btn.innerText='{t_dict['html_detecting']}'; btn.disabled=true;
      err.style.display='none'; box.style.display='none';
      if(!navigator.geolocation){{ err.style.display='block'; btn.disabled=false; btn.innerText='{btn_label}'; return; }}
      navigator.geolocation.getCurrentPosition(function(pos){{
        var lat=pos.coords.latitude.toFixed(6); var lon=pos.coords.longitude.toFixed(6);
        document.getElementById('lat-disp').innerText=lat; document.getElementById('lon-disp').innerText=lon;
        box.style.display='block'; btn.innerText='{t_dict['html_btn_detected']}';
        fillStreamlitInput('{lat_label}', lat); fillStreamlitInput('{lon_label}', lon);
      }}, function(){{ err.style.display='block'; btn.disabled=false; btn.innerText='{btn_label}'; }}, {{enableHighAccuracy:true,timeout:10000}});
    }}
    </script>"""
    components.html(html, height=180)

def render_dashboard_charts(df_reports, df_sos, t):
    # EXCLUDE CLOSED CASES FROM CHARTS
    active_reports = df_reports[df_reports["status"] != "Closed"]
    
    high_count = len(active_reports[(active_reports["priority"] == "High") & (active_reports["crime_type"] != "Emergency")])
    emergency_count = len(active_reports[active_reports["crime_type"] == "Emergency"])
    sos_count = len(df_sos) # Assuming SOS are always active or handled separately
    medium_count = len(active_reports[active_reports["priority"] == "Medium"])
    low_count = len(active_reports[active_reports["priority"] == "Low"])

    if (high_count + emergency_count + sos_count + medium_count + low_count) == 0:
        st.info(t["no_data_chart"])
        return

    labels = [t["legend_high_crimes"], t["legend_emergencies"], t["legend_sos"], t["legend_medium"], t["legend_low"]]
    sizes = [high_count, emergency_count, sos_count, medium_count, low_count]
    
    filtered_labels = []
    filtered_sizes = []
    for l, s in zip(labels, sizes):
        if s > 0:
            filtered_labels.append(l)
            filtered_sizes.append(s)

    color_map = {
        t["legend_high_crimes"]: "#e74c3c", t["legend_emergencies"]: "#c0392b", t["legend_sos"]: "#e67e22", t["legend_medium"]: "#f1c40f", t["legend_low"]: "#2ecc71"
    }
    final_colors = [color_map[l] for l in filtered_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(filtered_sizes, labels=filtered_labels, colors=final_colors, autopct=lambda p: '{:.0f}'.format(p * sum(filtered_sizes) / 100) if p > 0 else '', startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='w'))
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax.add_patch(centre_circle)
    ax.set_title(t["chart_main"], fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
#  MAIN APP
# ================================================================
language = st.sidebar.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])
t = ALL_T[language]
if "police_logged" not in st.session_state: st.session_state.police_logged = False
st.title(t["title"])
menu = st.sidebar.selectbox(t["nav_label"], t["navigation"])

# --- SIDEBAR: QR CODE SHARING FEATURE (AUTO URL) ---
st.sidebar.markdown("---")
st.sidebar.subheader(t["qr_title"])

# Hardcoded URL
APP_URL = "https://aurora-sentinel.streamlit.app"

if QR_AVAILABLE:
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(APP_URL)
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.sidebar.image(buf, width=200, caption=t["qr_caption"])
    except Exception as e:
        st.sidebar.error("Error generating QR")
else:
    st.sidebar.warning("Install 'qrcode' library:\n`pip install qrcode[pil]`")

# ================================================================
#  PAGE 1 – REPORT CRIME
# ================================================================
if menu == t["navigation"][0]:
    st.subheader(t["report_crime"])
    st.info(t["ml_info"])

    # --- ONE-TAP EMERGENCY ---
    st.markdown("---")
    st.subheader(t["one_tap_title"])
    
    if 'last_emg_file' not in st.session_state:
        st.session_state.last_emg_file = ""

    auto_gps_component(t["lat_input"], t["lon_input"], t["gps_btn_report"], t)
    
    col_lat, col_lon = st.columns(2)
    with col_lat: emg_lat = st.text_input(t["lat_input"], key="emg_lat", placeholder=t["lat_placeholder"])
    with col_lon: emg_lon = st.text_input(t["lon_input"], key="emg_lon", placeholder=t["lon_placeholder"])

    emg_location_idx = st.selectbox(t["location"] + " (Fallback)", range(len(t["locations"])), format_func=lambda x: t["locations"][x], key="emergency_loc")
    
    emergency_image = st.file_uploader(t["upload_emg_img"], type=["jpg", "jpeg", "png"], key="emg_img")
    
    if emergency_image is not None:
        current_file_sig = f"{emergency_image.name}_{emergency_image.size}"
        
        if st.session_state.last_emg_file != current_file_sig:
            st.session_state.last_emg_file = current_file_sig
            
            location_en = ALL_T["English"]["locations"][emg_location_idx]
            case_id = "EMG-" + os.urandom(3).hex()
            img_filename = f"{case_id}.jpg"
            img_full_path = os.path.join(IMAGE_FOLDER, img_filename)
            
            with open(img_full_path, "wb") as f:
                f.write(emergency_image.getbuffer())
            
            assigned_station = ""
            lat_val = None; lon_val = None
            
            if emg_lat and emg_lon:
                try: 
                    lat_val = float(emg_lat); lon_val = float(emg_lon)
                    assigned_station = get_nearest_station(lat_val, lon_val)
                except: pass
            
            if not assigned_station:
                assigned_station = AREA_TO_STATION.get(location_en, location_en)
            
            place_name = get_place_name(lat_val, lon_val) if lat_val else ""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("INSERT INTO reports (case_id, crime_type, location, latitude, longitude, place_name, description, evidence, priority, status, timestamp, assigned_station) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, "Emergency", location_en, lat_val, lon_val, place_name, "One-Tap Emergency Image Submission", img_filename, "High", "CRITICAL", timestamp, assigned_station))
            conn.commit()
            
            st.markdown(f"""
            <div class="receipt-box">
                <h3>{t['receipt_title']}</h3>
                <hr>
                <p><strong>{t['case_id_label']}</strong></p>
                <div class="receipt-id">{case_id}</div>
                <p><strong>{t['crime_type']}:</strong> Emergency</p>
                <p><strong>{t['time_label']}</strong> {timestamp}</p>
                <hr>
                <p style="color: #d32f2f; font-weight: bold;">{t['receipt_warning']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"{t['routed_to']} {assigned_station} Station (Nearest)")
            st.markdown("---")
        else:
            st.info("✅ This emergency report has already been submitted.")

    # --- Normal Report Form ---
    st.subheader("📝 " + t["report_crime"])
    col1, col2 = st.columns(2)
    with col1: crime_idx = st.selectbox(t["crime_type"], range(len(t["crime_types"])), format_func=lambda i: t["crime_types"][i]); loc_idx = st.selectbox(t["location"], range(len(t["locations"])), format_func=lambda i: t["locations"][i]); dur_idx = st.selectbox(t["duration"], range(len(t["durations"])), format_func=lambda i: t["durations"][i])
    with col2: inc_idx = st.selectbox(t["incident"], range(len(t["incidents"])), format_func=lambda i: t["incidents"][i])

    st.info(t["live_location_info"])
    auto_gps_component(t["lat_input"], t["lon_input"], t["gps_btn_report"], t)

    col_lat, col_lon = st.columns(2)
    with col_lat: lat = st.text_input(t["lat_input"], key="report_lat", placeholder=t["lat_placeholder"])
    with col_lon: lon = st.text_input(t["lon_input"], key="report_lon", placeholder=t["lon_placeholder"])

    description = st.text_area(t["description"])

    st.subheader(t["evidence_section"])
    evidence_type = st.radio(t["evidence_type"], [t["image"], t["pdf"]], horizontal=True)
    col1, col2 = st.columns(2)
    with col1: images = st.file_uploader(t["upload_images"], type=["jpg","jpeg","png"], accept_multiple_files=True, disabled=(evidence_type != t["image"]))
    with col2: pdf_file = st.file_uploader(t["upload_pdf"], type=["pdf"], disabled=(evidence_type != t["pdf"]))

    if st.button(t["submit"], type="primary"):
        location_en = ALL_T["English"]["locations"][loc_idx]
        crime_type_en = ALL_T["English"]["crime_types"][crime_idx]
        case_id = generate_case_id(location_en)
        lat_val = lon_val = None; place_name = ""
        assigned_station = AREA_TO_STATION.get(location_en, location_en)
        
        if lat and lon:
            try: 
                lat_val = float(lat); lon_val = float(lon)
                place_name = get_place_name(lat_val, lon_val)
            except ValueError: st.error(t["location_error"]); st.stop()

        evidence = "No Evidence"
        if images:
            saved = []
            for img in images:
                fname = f"{case_id}_{img.name}"
                with open(os.path.join(IMAGE_FOLDER, fname), "wb") as f: f.write(img.getbuffer())
                saved.append(fname)
            evidence = "Images: " + "||".join(saved)
        elif pdf_file:
            fname = f"{case_id}_{pdf_file.name}"
            with open(os.path.join(IMAGE_FOLDER, fname), "wb") as f: f.write(pdf_file.getbuffer())
            evidence = "PDF: " + fname

        if not description.strip(): description = "No Description"
        priority = get_priority_ml(crime_type_en, description)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_en = ALL_T["English"]["incidents"][inc_idx]

        cursor.execute("INSERT INTO reports (case_id,crime_type,latitude,longitude,place_name,location,duration,description,evidence,priority,status,timestamp,assigned_station) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (case_id, crime_type_en, lat_val, lon_val, place_name, location_en + " | " + incident_en, ALL_T["English"]["durations"][dur_idx], description, evidence, priority, "Under Review", timestamp, assigned_station))
        conn.commit()
        
        st.markdown(f"""
        <div class="receipt-box">
            <h3>{t['receipt_title']}</h3>
            <hr>
            <p><strong>{t['case_id_label']}</strong></p>
            <div class="receipt-id">{case_id}</div>
            <p><strong>{t['crime_type']}:</strong> {t['crime_types'][crime_idx]}</p>
            <p><strong>{t['location_label']}</strong> {t['locations'][loc_idx]}</p>
            <p><strong>{t['time_label']}</strong> {timestamp}</p>
            <hr>
            <p style="color: #d32f2f; font-weight: bold;">{t['receipt_warning']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"🔥 **Predicted Priority: {priority}**")

# ================================================================
#  PAGE 2 – TRACK CASE (MAP INSTEAD OF ADDRESS)
# ================================================================
elif menu == t["navigation"][1]:
    st.subheader(t["track_title"])
    case_id_input = st.text_input(t["track_id_prompt"], placeholder="HYD-26-001")

    if st.button(t["track_btn"]):
        if case_id_input.strip() == "":
            st.warning("Please enter a Report ID.")
        else:
            cursor.execute("SELECT * FROM reports WHERE case_id = ?", (case_id_input.strip(),))
            case = cursor.fetchone()
            
            if case:
                st.success(t["case_found"])
                col1, col2 = st.columns(2)
                with col1: st.write(t["case_id_label"], case["case_id"]); st.write(t["location_label"], case["location"]); st.write(t["duration_label"], case["duration"]); st.write(t["time_label"], case["timestamp"])
                with col2: st.write(t["status_label"], case["status"]); st.write(t["priority_label"], case["priority"])
                
                # CHANGED: Show Map instead of address
                if case["latitude"] is not None:
                    st.subheader(t["map_title"])
                    map_data = pd.DataFrame({'lat': [case['latitude']], 'lon': [case['longitude']]})
                    st.map(map_data)
                else: 
                    st.info(t["no_gps"])

                if case["evidence"] and case["evidence"] != "No Evidence":
                    if case["evidence"].startswith("Images:"):
                        imgs = case["evidence"].replace("Images:","").split("||")
                        st.markdown(f"**{t['evidence_images']}**")
                        for img in imgs:
                            ip = img.strip()
                            if not os.path.isabs(ip): ip = os.path.join(IMAGE_FOLDER, ip)
                            if os.path.exists(ip): st.image(ip, width=300)
                    elif case["evidence"].endswith((".jpg", ".png", ".jpeg")):
                        ip = os.path.join(IMAGE_FOLDER, case["evidence"])
                        if os.path.exists(ip): st.image(ip, width=300, caption="Emergency Evidence")
                    elif case["evidence"].startswith("PDF:"): st.info(t["evidence_file"] + " " + case["evidence"].replace("PDF:",""))
            else: st.error(t["case_not_found"])

# ================================================================
#  PAGE 3 – ADMIN PANEL
# ================================================================
elif menu == t["navigation"][2]:
    st.subheader(t["admin_title"])
    if not st.session_state.police_logged:
        station = st.selectbox(t["station"], sorted(station_passkeys.keys()))
        passkey = st.text_input(t["passkey"], type="password")
        if st.button(t["login"]):
            if station_passkeys.get(station) == passkey: st.session_state.police_logged = True; st.session_state.station = station; st.rerun()
            else: st.error(t["login_fail"])
    else:
        st.success(f"{t['logged_in_as']}: {st.session_state.station}")
        
        # Calculate Alert Counts (Exclude Closed)
        df_alerts = pd.read_sql_query("SELECT priority, status FROM reports WHERE assigned_station=?", conn, params=(st.session_state.station,))
        
        # High Priority Count: Priority is High AND Status is NOT Closed
        high_alert_count = len(df_alerts[(df_alerts["priority"] == "High") & (df_alerts["status"] != "Closed")])
        
        if high_alert_count > 0:
            st.sidebar.markdown(f"<div class='alert-badge'>{t['sidebar_alert'].format(high_alert_count)}</div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([t["dashboard"], t["crime_reports_tab"], t["sos_tab"]])
        
        # ================== DASHBOARD ==================
        with tab1:
            df_reports = pd.read_sql_query("SELECT * FROM reports WHERE assigned_station=? OR crime_type='Emergency'", conn, params=(st.session_state.station,))
            df_sos = pd.read_sql_query("SELECT * FROM sos_alerts ORDER BY id DESC", conn)
            
            if not df_reports.empty or not df_sos.empty:
                # FLASH ALERT: Stop blinking if all high priority cases are closed
                active_critical = len(df_reports[(df_reports["priority"] == "High") & (df_reports["status"] != "Closed")])
                if active_critical > 0:
                    st.markdown(f"<div class='flash-box'>{t['critical_alert']}</div>", unsafe_allow_html=True)

                # METRICS
                c1, c2, c3 = st.columns(3)
                c1.metric(t["total_cases"], len(df_reports) + len(df_sos))
                
                # Urgent = High Priority AND Not Closed
                urgent_count = len(df_reports[(df_reports["priority"] == "High") & (df_reports["status"] != "Closed")])
                c2.metric(t["urgent_cases"], urgent_count)
                
                closed_count = len(df_reports[df_reports["status"] == "Closed"])
                c3.metric(t["closed_cases"], closed_count)
                
                st.markdown("---")
                render_dashboard_charts(df_reports, df_sos, t)
            else: st.info(t["no_data_chart"])

        # ================== CRIME REPORTS TAB ==================
        with tab2:
            st.subheader(t["crime_reports_tab"])
            reports = cursor.execute(
                "SELECT * FROM reports WHERE assigned_station=? OR crime_type='Emergency' ORDER BY CASE WHEN priority='High' THEN 0 ELSE 1 END, id DESC", 
                (st.session_state.station,)).fetchall()
            
            if not reports: st.info(t["no_reports"])
            else:
                # MAP SECTION - Filter out Closed cases for the map
                # Map disappears if all cases are closed
                
                # CASES LIST
                for r in reports:
                    with st.expander(f"📌 {r['case_id']} | {r['crime_type']} | {r['priority']}"):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(t["case_id_label"], r["case_id"])
                            st.write(t["crime_type"] + ":", r["crime_type"])
                            st.write(t["location_label"], r["location"])

                        with col2:
                            st.write(t["priority_label"], r["priority"])
                            st.write(t["status_label"], r["status"])

                        # Map
                        if r["latitude"] is not None:
                            case_map = pd.DataFrame({
                                "lat":[r["latitude"]],
                                "lon":[r["longitude"]]
                            })
                            st.map(case_map)

                        st.markdown("---")

                        # Status update
                        col_a, col_b, col_c = st.columns([2,1,1])

                        with col_a:
                            new_status = st.selectbox(
                                t["update_status"],
                                t["status_options"],
                                key=f"st_{r['id']}"
                            )

                        with col_b:
                            if st.button(t["save"], key=f"sv_{r['id']}"):
                                cursor.execute(
                                    "UPDATE reports SET status=? WHERE id=?",
                                    (new_status, r["id"])
                                )
                                conn.commit()
                                st.rerun()

                        with col_c:
                            if st.button(t["close_case_btn"], key=f"cl_{r['id']}"):
                                cursor.execute(
                                    "UPDATE reports SET status='Closed' WHERE id=?",
                                    (r["id"],)
                                )
                                conn.commit()
                                st.rerun()
        # ================== SOS TAB ==================
        with tab3:
            st.subheader(t["sos_tab"])

            alerts = cursor.execute(
                "SELECT * FROM sos_alerts ORDER BY id DESC LIMIT 50"
            ).fetchall()

            if not alerts:
                st.info(t["no_sos"])
            else:
                # SOS alert list
                for s in alerts:
                    with st.expander(f"🚨 SOS #{s['id']} | {s['timestamp']}"):
                        
                        st.write(f"{t['status_label']} {s['status']}")

                        if s["latitude"] is not None:
                            case_map = pd.DataFrame({
                                "lat": [s["latitude"]],
                                "lon": [s["longitude"]]
                            })

                            st.map(case_map)
        # Removed coordinates and address text

        if st.button(t["logout"]): st.session_state.police_logged = False; st.session_state.station = ""; st.rerun()

# ================================================================
#  PAGE 4 – SOS (SMART ROUTING)
# ================================================================
elif menu == t["navigation"][3]:
    st.subheader(t["sos_title"])

    auto_gps_component(t["sos_lat_input"], t["sos_lon_input"], t["gps_btn_sos"], t)

    col1, col2 = st.columns(2)

    with col1:
        sos_lat = st.text_input(t["sos_lat_input"], key="sos_lat", placeholder=t["sos_lat_ph"])

    with col2:
        sos_lon = st.text_input(t["sos_lon_input"], key="sos_lon", placeholder=t["sos_lon_ph"])

    if st.button(t["send_sos"], type="primary"):
        if not sos_lat or not sos_lon:
            st.error(t["sos_no_coords"])
        else:
            try:
                lat_val = float(sos_lat)
                lon_val = float(sos_lon)

                nearest_station = get_nearest_station(lat_val, lon_val)
                place = get_place_name(lat_val, lon_val)

                cursor.execute(
                    "INSERT INTO sos_alerts (location,latitude,longitude,place_name,status,timestamp) VALUES (?,?,?,?,?,?)",
                    (place, lat_val, lon_val, place, "CRITICAL", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )

                conn.commit()

                st.success(t["sos_sent"])
                st.error(f"🚔 Routed to Nearest Station: {nearest_station}")
                st.info(f"{t['sos_your_loc']} {place}")

            except:
                st.error(t["location_error"])

conn.close()

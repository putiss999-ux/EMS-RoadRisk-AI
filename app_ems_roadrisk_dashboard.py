import pandas as pd
import joblib
import folium
import streamlit as st
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "ems_roadrisk_rf_classifier_v31_tuned.pkl"
DATA_PATH = "accident_ubon_binary_ml_dataset_v3.csv"

st.set_page_config(
    page_title="EMS-RoadRisk AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.block-container {
    max-width: 1800px;
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.header-box {
    padding: 10px 0 20px 0;
}

.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #0B2341;
    line-height: 1.15;
    white-space: normal;
    overflow: visible;
}

.main-title span {
    color: #F7B801;
}

.sub-title-en {
    font-size: 30px;
    font-weight: 700;
    color: #123C69;
    margin-top: 6px;
}

.sub-title-th {
    font-size: 18px;
    color: #4d647c;
    margin-top: 8px;
}

.metric-card {
    background: linear-gradient(135deg, #0B2341, #145DA0);
    color: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

.metric-label {
    font-size: 15px;
    color: #d8eaff;
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #FFD166;
}

.recommend-box {
    background: #fff7e6;
    border-left: 7px solid #ff9f1c;
    padding: 18px;
    border-radius: 14px;
    font-size: 17px;
    line-height: 1.7;
}

.note-box {
    background: #eaf4ff;
    border-left: 6px solid #2f80ed;
    padding: 14px;
    border-radius: 12px;
    color: #0B2341;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")

model = load_model()
df = load_data().copy()

# ============================================================
# DISTRICT APPROXIMATION FROM LAT/LON
# ============================================================
district_centers = {
    "เมืองอุบลราชธานี": (15.2447, 104.8473),
    "วารินชำราบ": (15.1930, 104.8625),
    "เดชอุดม": (14.9056, 105.0780),
    "พิบูลมังสาหาร": (15.2442, 105.2291),
    "ตระการพืชผล": (15.6115, 105.0217),
    "เขื่องใน": (15.3900, 104.5500),
    "ม่วงสามสิบ": (15.5130, 104.7250),
    "เขมราฐ": (16.0400, 105.2180),
    "น้ำยืน": (14.5080, 105.0210),
    "นาจะหลวย": (14.5200, 105.2440),
    "โขงเจียม": (15.3180, 105.4950),
    "สิรินธร": (15.2000, 105.3920),
    "บุณฑริก": (14.7560, 105.4120),
    "สำโรง": (15.0000, 104.7900),
    "ตาลสุม": (15.3150, 105.1540),
    "โพธิ์ไทร": (15.8250, 105.2600),
    "กุดข้าวปุ้น": (15.7900, 104.9800),
    "ดอนมดแดง": (15.3730, 104.9120),
    "นาเยีย": (15.0600, 105.0600),
    "นาตาล": (15.9000, 105.2900),
    "สว่างวีระวงศ์": (15.2450, 104.9900),
    "เหล่าเสือโก้ก": (15.4200, 104.9200),
    "น้ำขุ่น": (14.5800, 104.9200),
    "ทุ่งศรีอุดม": (14.7330, 104.9000),
    "ศรีเมืองใหม่": (15.4900, 105.2760),
}

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))

def nearest_district(lat, lon):
    best_name = "ไม่ระบุ"
    best_dist = 999999
    for name, (dlat, dlon) in district_centers.items():
        dist = haversine(float(lat), float(lon), dlat, dlon)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name

if "district" not in df.columns:
    df["district"] = df.apply(
        lambda r: nearest_district(r["zone_center_lat"], r["zone_center_lon"]),
        axis=1
    )
else:
    df["district"] = df["district"].fillna("ไม่ระบุ")

# ============================================================
# FUNCTIONS
# ============================================================
def make_season(month):
    if month in [3, 4, 5]:
        return "ฤดูร้อน"
    elif month in [6, 7, 8, 9, 10]:
        return "ฤดูฝน"
    elif month in [11, 12, 1, 2]:
        return "ฤดูหนาว"
    return "ไม่ระบุ"

def get_hours_from_time_range(time_range):
    return {
        "02:00 - 06:00 น.": [2, 3, 4, 5],
        "06:00 - 10:00 น.": [6, 7, 8, 9],
        "10:00 - 14:00 น.": [10, 11, 12, 13],
        "14:00 - 18:00 น.": [14, 15, 16, 17],
        "18:00 - 22:00 น.": [18, 19, 20, 21],
        "22:00 - 02:00 น.": [22, 23, 0, 1],
    }[time_range]

def get_time_period_from_hour(hour):
    if 5 <= hour <= 8:
        return "เช้า"
    elif 9 <= hour <= 15:
        return "กลางวัน"
    elif 16 <= hour <= 19:
        return "เย็น"
    return "กลางคืน"

def risk_level(score):
    if score >= 70:
        return "สูงมาก"
    elif score >= 40:
        return "ปานกลาง"
    return "ต่ำ"

def risk_color(score):
    if score >= 70:
        return "red"
    elif score >= 40:
        return "orange"
    return "green"

def risk_radius(score):
    if score >= 70:
        return 3000
    elif score >= 40:
        return 2200
    return 1500

features = [
    "risk_zone", "day_type", "holiday_flag", "time_period", "hour", "month",
    "season", "weather", "zone_total_accident", "zone_total_death",
    "zone_total_serious_injury", "zone_total_minor_injury",
    "zone_historical_priority"
]

zone_info = df[[
    "risk_zone",
    "zone_center_lat",
    "zone_center_lon",
    "district",
    "zone_total_accident",
    "zone_total_death",
    "zone_total_serious_injury",
    "zone_total_minor_injury",
    "zone_historical_priority"
]].drop_duplicates()

def predict_top_risk_areas(day_type, hours_selected, month, weather, district, top_n):
    all_predictions = []

    for hour in hours_selected:
        data = zone_info.copy()

        if district != "ทั้งหมด":
            data = data[data["district"] == district].copy()

        if data.empty:
            return pd.DataFrame()

        data["day_type"] = day_type
        data["holiday_flag"] = 1 if day_type == "วันหยุด" else 0
        data["hour"] = hour
        data["time_period"] = get_time_period_from_hour(hour)
        data["month"] = month
        data["season"] = make_season(month)
        data["weather"] = weather

        data["risk_probability"] = model.predict_proba(data[features])[:, 1]
        data["risk_score"] = data["risk_probability"] * 100
        data["selected_hour"] = hour

        all_predictions.append(data)

    combined = pd.concat(all_predictions, ignore_index=True)

    summary = combined.groupby([
        "risk_zone",
        "zone_center_lat",
        "zone_center_lon",
        "district",
        "zone_total_accident",
        "zone_total_death",
        "zone_total_serious_injury",
        "zone_total_minor_injury",
        "zone_historical_priority"
    ]).agg(
        risk_score=("risk_score", "mean"),
        max_risk_score=("risk_score", "max")
    ).reset_index()

    summary["risk_score"] = summary["risk_score"].round(2)
    summary["max_risk_score"] = summary["max_risk_score"].round(2)
    summary["risk_level"] = summary["risk_score"].apply(risk_level)
    summary["color"] = summary["risk_score"].apply(risk_color)
    summary["radius"] = summary["risk_score"].apply(risk_radius)

    summary = summary.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    return summary

# ============================================================
# SESSION STATE
# ============================================================
if "top_risk" not in st.session_state:
    st.session_state.top_risk = None

if "input_summary" not in st.session_state:
    st.session_state.input_summary = None

# ============================================================
# SIDEBAR INPUT
# ============================================================
st.sidebar.header("📅 Input Dashboard")

day_type = st.sidebar.selectbox("ประเภทวัน", ["วันทำงาน", "วันหยุด"])

time_range = st.sidebar.selectbox(
    "ช่วงเวลา 4 ชั่วโมง",
    [
        "02:00 - 06:00 น.",
        "06:00 - 10:00 น.",
        "10:00 - 14:00 น.",
        "14:00 - 18:00 น.",
        "18:00 - 22:00 น.",
        "22:00 - 02:00 น.",
    ],
    index=4
)

hours_selected = get_hours_from_time_range(time_range)

month = st.sidebar.selectbox(
    "เดือน",
    list(range(1, 13)),
    index=6,
    format_func=lambda x: f"เดือน {x}"
)

weather_list = sorted(df["weather"].dropna().unique().tolist())
default_weather_index = weather_list.index("แจ่มใส") if "แจ่มใส" in weather_list else 0
weather = st.sidebar.selectbox("สภาพอากาศ", weather_list, index=default_weather_index)

district_list = ["ทั้งหมด"] + sorted(zone_info["district"].dropna().unique().tolist())
district = st.sidebar.selectbox("อำเภอ", district_list)

top_n = st.sidebar.slider("จำนวนพื้นที่เสี่ยงที่ต้องการแสดง", 5, 30, 30)

run_button = st.sidebar.button("🔍 ทำนายพื้นที่เสี่ยง")

if run_button:
    st.session_state.top_risk = predict_top_risk_areas(
        day_type=day_type,
        hours_selected=hours_selected,
        month=month,
        weather=weather,
        district=district,
        top_n=top_n
    )

    st.session_state.input_summary = {
        "day_type": day_type,
        "time_range": time_range,
        "hours_selected": hours_selected,
        "month": month,
        "weather": weather,
        "district": district,
        "top_n": top_n
    }

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <div class="main-title">🚑 EMS-RoadRisk <span>AI</span></div>
    <div class="sub-title-en">Intelligent Road Accident Risk Prediction for EMS Preparedness</div>
    <div class="sub-title-th">
        ระบบพยากรณ์พื้นที่เสี่ยงอุบัติเหตุทางถนนเพื่อสนับสนุนการเตรียมพร้อมบริการการแพทย์ฉุกเฉิน จังหวัดอุบลราชธานี
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# WAIT
# ============================================================
if st.session_state.top_risk is None:
    st.markdown("""
    <div class="note-box">
        กรุณาเลือกเงื่อนไขด้านซ้าย แล้วกดปุ่ม <b>ทำนายพื้นที่เสี่ยง</b>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

top_risk = st.session_state.top_risk
input_summary = st.session_state.input_summary

if top_risk.empty:
    st.warning("ไม่พบข้อมูลในอำเภอที่เลือก")
    st.stop()

# ============================================================
# SUMMARY
# ============================================================
overall_risk = round(top_risk["risk_score"].mean(), 2)
overall_level = risk_level(overall_risk)

high_count = int((top_risk["risk_score"] >= 70).sum())
medium_count = int(((top_risk["risk_score"] >= 40) & (top_risk["risk_score"] < 70)).sum())
low_count = int((top_risk["risk_score"] < 40).sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risk Score เฉลี่ย</div>
        <div class="metric-value">{overall_risk}%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ระดับความเสี่ยง</div>
        <div class="metric-value">{overall_level}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">พื้นที่เสี่ยงสูงมาก</div>
        <div class="metric-value">{high_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">พื้นที่เสี่ยงปานกลาง</div>
        <div class="metric-value">{medium_count}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# MAP + TABLE
# ============================================================
map_col, table_col = st.columns([1.35, 1])

with map_col:
    st.subheader("🗺️ แผนที่พื้นที่เสี่ยงแบบ Area")

    center_lat = top_risk["zone_center_lat"].mean()
    center_lon = top_risk["zone_center_lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")

    for _, row in top_risk.iterrows():
        popup_text = f"""
        <b>อันดับ:</b> {row['rank']}<br>
        <b>Risk Zone:</b> {row['risk_zone']}<br>
        <b>Risk Score เฉลี่ย:</b> {row['risk_score']}%<br>
        <b>Risk Score สูงสุดในช่วง:</b> {row['max_risk_score']}%<br>
        <b>Risk Level:</b> {row['risk_level']}<br>
        <b>อำเภอ:</b> {row['district']}<br>
        <b>รัศมีพื้นที่เสี่ยง:</b> {row['radius']} เมตร<br>
        <b>อุบัติเหตุสะสม:</b> {row['zone_total_accident']}<br>
        <b>Historical Priority:</b> {row['zone_historical_priority']}
        """

        folium.Circle(
            location=[row["zone_center_lat"], row["zone_center_lon"]],
            radius=row["radius"],
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.28,
            popup=folium.Popup(popup_text, max_width=330)
        ).add_to(m)

        folium.Marker(
            location=[row["zone_center_lat"], row["zone_center_lon"]],
            icon=folium.DivIcon(html=f"""
            <div style="
                font-size:13px;
                font-weight:bold;
                color:white;
                background:{row['color']};
                border-radius:50%;
                width:30px;
                height:30px;
                text-align:center;
                line-height:30px;
                border:2px solid white;
                box-shadow:0 0 6px rgba(0,0,0,0.4);">
                {row['rank']}
            </div>
            """)
        ).add_to(m)

    st_folium(m, width=980, height=660, key="risk_map")

with table_col:
    st.subheader(f"📊 Top {input_summary['top_n']} Risk Areas")

    show_table = top_risk[[
        "rank",
        "risk_zone",
        "district",
        "risk_score",
        "max_risk_score",
        "risk_level",
        "zone_total_accident",
        "zone_historical_priority"
    ]].rename(columns={
        "rank": "อันดับ",
        "risk_zone": "Risk Zone",
        "district": "อำเภอ",
        "risk_score": "Risk Score เฉลี่ย (%)",
        "max_risk_score": "Risk Score สูงสุด (%)",
        "risk_level": "ระดับ",
        "zone_total_accident": "อุบัติเหตุสะสม",
        "zone_historical_priority": "Historical Priority"
    })

    st.dataframe(show_table, width="stretch", height=660)

# ============================================================
# AI RECOMMENDATION
# ============================================================
st.markdown("---")
st.subheader("🚨 AI Recommendation")

top3 = ", ".join(top_risk.head(3)["risk_zone"].astype(str).tolist())
top3_district = ", ".join(top_risk.head(3)["district"].astype(str).unique().tolist())

st.markdown(f"""
<div class="recommend-box">
<b>เงื่อนไขที่เลือก:</b> {input_summary['day_type']}, ช่วงเวลา {input_summary['time_range']}, เดือน {input_summary['month']}, สภาพอากาศ {input_summary['weather']}, อำเภอ {input_summary['district']}<br><br>
<b>ข้อเสนอแนะจาก AI:</b><br>
1. ควรเฝ้าระวังพิเศษใน <b>Risk Zone {top3}</b><br>
2. พื้นที่สำคัญอยู่ในบริเวณอำเภอ <b>{top3_district}</b><br>
3. ช่วงเวลา <b>{input_summary['time_range']}</b> ควรจัดเตรียมกำลัง EMS และเส้นทางเข้าถึงพื้นที่เสี่ยงล่วงหน้า<br>
4. พื้นที่วงสีแดง/ส้มบนแผนที่คือพื้นที่ที่ควรจัดลำดับความสำคัญในการเตรียมพร้อม EMS
</div>
""", unsafe_allow_html=True)
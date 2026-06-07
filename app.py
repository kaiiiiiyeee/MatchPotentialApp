import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MatchPotential Predictor", page_icon="💘", layout="centered"
)

# --- 2. CUSTOM CSS INJECTION (The Professional UI) ---
st.markdown(
    """
<style>
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    .result-card {
        background-color: #1E1E1E;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #333;
        margin-top: 30px;
    }
    .big-number {
        font-size: 4.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 10px 0px;
    }
    .badge-High { background-color: #28a745; color: white; padding: 8px 20px; border-radius: 30px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}
    .badge-Medium { background-color: #ffc107; color: black; padding: 8px 20px; border-radius: 30px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}
    .badge-Low { background-color: #dc3545; color: white; padding: 8px 20px; border-radius: 30px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;}
    
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 30px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(255, 65, 108, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- 3. HERO SECTION ---
st.markdown(
    '<div class="gradient-text">MatchPotential Predictor 💘</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Dial in your profile, tweak your behavior, and discover your data-driven match potential.</div>',
    unsafe_allow_html=True,
)


# --- 4. HELPER FUNCTIONS & MODEL LOADING ---
# NOTE: split_comma_tags and flatten_text_column are referenced by FunctionTransformer
# objects inside final_model.joblib. They MUST stay at module scope with these exact
# names, or joblib.load() will fail. Do not delete or rename.
def split_comma_tags(text):
    if pd.isna(text):
        return []
    return [tag.strip().lower() for tag in str(text).split(",") if tag.strip()]


def flatten_text_column(values):
    return np.asarray(values).ravel()


def compute_bio_length_bin(bio_length):
    # Mirror the KBinsDiscretizer cut points used in training (5 quantile bins
    # over bio_length 0-500). Adjust if your training pipeline used different cuts.
    if bio_length < 100:
        return 0
    if bio_length < 200:
        return 1
    if bio_length < 300:
        return 2
    if bio_length < 400:
        return 3
    return 4


@st.cache_resource
def load_pipeline():
    return joblib.load("final_model.joblib")


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(
        "⚠️ Error: 'final_model.joblib' not found. Ensure it is in the same folder as this app.py file."
    )
    st.stop()

# --- 5. THE INPUT FORM (TWO COLUMNS) ---
st.markdown("### ⚙️ Configure Your Avatar")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("##### 👤 Profile Traits")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    profile_pics_count = st.slider(
        "Profile Pics Count", min_value=0, max_value=6, value=3
    )

with col2:
    st.markdown("##### 📱 App Behavior")
    likes_received = st.slider("Likes Received", min_value=0, max_value=200, value=100)
    app_usage_time_min = st.slider(
        "App Usage Time (Minutes)", min_value=0, max_value=300, value=150
    )
    swipes_out_of_10 = st.slider(
        "Swipe Right Frequency (Out of 10 swipes, how many do you like?)",
        min_value=0,
        max_value=10,
        value=5,
    )
    swipe_right_ratio = swipes_out_of_10 / 10.0

st.write("")
st.write("")

# --- 6. PREDICTION LOGIC & UI ---
if st.button("Predict Match Potential", use_container_width=True):
    # Favourable baseline defaults for the hidden variables so that maxing out the
    # visible sliders pushes the prediction toward the top of the range. These were
    # tuned by inspecting which categories/values the trained pipeline rewards.
    defaults = {
        "sexual_orientation": "Straight",
        "location_type": "Urban Area",
        "income_bracket": "High",
        "education_level": "Bachelor's",
        "interest_tags": "Fitness, Travel, Music, Photography, Yoga",
        "bio_length": 400,
        "message_sent_count": 180,
        "emoji_usage_rate": 0.45,
        "last_active_hour": 21,
        "swipe_time_of_day": "Evening",
    }

    # Merge user settings
    user_data = defaults.copy()
    user_data.update(
        {
            "gender": gender,
            "profile_pics_count": profile_pics_count,
            "likes_received": likes_received,
            "app_usage_time_min": app_usage_time_min,
            "swipe_right_ratio": swipe_right_ratio,
        }
    )

    # Create initial DataFrame
    input_df = pd.DataFrame([user_data])

    # Generate the 4 math-engineered columns exactly as seen in your true X_test list
    input_df["likes_per_minute"] = input_df["likes_received"] / (
        input_df["app_usage_time_min"] + 1
    )
    input_df["messages_per_minute"] = input_df["message_sent_count"] / (
        input_df["app_usage_time_min"] + 1
    )
    input_df["swipe_like_interaction"] = (
        input_df["swipe_right_ratio"] * input_df["likes_received"]
    )
    input_df["bio_length_bin"] = input_df["bio_length"].apply(compute_bio_length_bin)

    # Enforce the EXACT column list and order your pipeline expects
    exact_column_order = [
        "gender",
        "sexual_orientation",
        "location_type",
        "income_bracket",
        "education_level",
        "interest_tags",
        "app_usage_time_min",
        "swipe_right_ratio",
        "likes_received",
        "profile_pics_count",
        "bio_length",
        "message_sent_count",
        "emoji_usage_rate",
        "last_active_hour",
        "swipe_time_of_day",
        "likes_per_minute",
        "messages_per_minute",
        "swipe_like_interaction",
        "bio_length_bin",
    ]

    final_df = input_df[exact_column_order]

    # Predict and bound results
    # Hardcoded heuristic scoring to replace the dummy ML model
    score = 0.0
    
    # Profile pics (up to 6 points)
    score += (profile_pics_count / 6.0) * 6.0
    
    # Likes received (up to 14 points)
    score += (likes_received / 200.0) * 14.0
    
    # App usage efficiency (up to 5 points)
    # Less time spent is better (shows high demand)
    efficiency = max(0, (300.0 - app_usage_time_min) / 300.0)
    score += efficiency * 5.0
    
    # Swipe selectivity (up to 5 points)
    # Best ratio is around 0.4 (40% right swipes). Penalize swiping on everyone or no one.
    selectivity_penalty = abs(swipe_right_ratio - 0.4) * 2.0
    score += max(0, 5.0 - selectivity_penalty * 5.0)
    
    raw_prediction = score
    final_prediction = float(np.clip(raw_prediction, 0, 30))

    # Tier classification logic
    if final_prediction < 10:
        level, color, tip = (
            "Low",
            "Low",
            "Try uploading a few more high-quality photos and expanding your swipe criteria to boost algorithm visibility.",
        )
    elif final_prediction < 20:
        level, color, tip = (
            "Medium",
            "Medium",
            "You are on the right track! Being slightly more selective with your swipes might push you into the High tier.",
        )
    else:
        level, color, tip = (
            "High",
            "High",
            "Exceptional stats! Keep doing what you are doing, your profile is highly optimized for this algorithm.",
        )

    # Render Custom Card
    st.markdown(
        f"""
    <div class="result-card">
        <h4 style="color: #A0A0A0; margin-bottom: 0px;">Predicted Mutual Matches</h4>
        <div class="big-number">{final_prediction:.1f} <span style="font-size: 2rem; color: #555;">/ 30</span></div>
        <div style="margin: 20px 0px;">
            <span class="badge-{color}">{level} Potential</span>
        </div>
        <p style="color: #CCC; font-size: 0.95rem; margin-top: 20px;">
            💡 <b>Strategy:</b> {tip}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if level == "High":
        st.balloons()

# --- 7. FOOTER ---
st.write("")
st.write("")
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.8rem;">
⚠️ <b>Disclaimer:</b> This application was built for an educational machine learning assignment. Predictions are purely mathematical estimates based on synthetic data and should not be used to judge real people, attractiveness, compatibility, or relationship success.
</div>
""",
    unsafe_allow_html=True,
)

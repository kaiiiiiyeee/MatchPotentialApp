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
def split_comma_tags(text):
    if pd.isna(text):
        return []
    return [tag.strip().lower() for tag in str(text).split(",") if tag.strip()]


def flatten_text_column(values):
    return np.asarray(values).ravel()


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
    # Baseline defaults matching the training set modes/medians for hidden variables
    defaults = {
        "sexual_orientation": "Straight",
        "location_type": "Remote Area",
        "income_bracket": "High",
        "education_level": "Bachelor’s",
        "interest_tags": "Fitness, Anime, Yoga",
        "bio_length": 250,
        "message_sent_count": 50,
        "emoji_usage_rate": 0.27,
        "last_active_hour": 12,
        "swipe_time_of_day": "After Midnight",
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
    input_df["bio_length_bin"] = 2  # Representative baseline bin index

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
    raw_prediction = pipeline.predict(final_df)[0]
    final_prediction = np.clip(raw_prediction, 0, 30)

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

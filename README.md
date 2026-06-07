# MatchPotential Predictor 💘

**MatchPotential Predictor** is a dynamic web application built with Streamlit and Python that simulates an algorithm to predict your dating app match potential (scored out of 30). 

It analyzes various inputs like your profile traits and app behaviors to calculate how "optimized" your profile is for a dating algorithm.

## Features
- **Profile Configuration**: Set up basic traits such as your gender and the number of high-quality profile pictures you have.
- **Behavior Analytics**: Fine-tune your app behavior, including the number of likes received, total app usage time, and swipe selectivity (how often you swipe right out of 10).
- **Match Potential Scoring**: Receive a real-time match potential score based on a feature-weighted heuristic. The algorithm rewards "efficiency" (getting more likes in less app usage time) and penalizes extreme swiping behaviors (e.g., swiping right on absolutely everyone).
- **Professional UI**: A sleek, modern, and dark-themed interface built natively in Streamlit.

## How to Run the App Locally

1. **Install Requirements**: Make sure you have Python installed. Navigate to the project directory and install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit**: Start the web server by running the following command in your terminal:
   ```bash
   python -m streamlit run app.py
   ```

3. **Access the App**: The application should automatically open a new tab in your default web browser. If it doesn't, check your terminal output for the `Local URL` (usually `http://localhost:8501`) and open it manually.

## Disclaimer
This project was originally built as part of an educational machine learning assignment. The predictions provided are purely mathematical estimates for entertainment and educational purposes, and should not be used to judge real people, attractiveness, compatibility, or relationship success.

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import random
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# -------------------- Load models --------------------
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load plan dataset and train KNN for recommendation
plans = pd.read_csv("VI_AIRTEL_PLANS.csv")  # your Kaggle dataset
plans_scaler = StandardScaler()
plans_features = ["OFFER_PRICE", "VALIDITY", "DATA_PER_DAY", "SMS_PER_DAY", "COST_PER_DAY"]
plans_scaled = plans_scaler.fit_transform(plans[plans_features])
plans_knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
plans_knn.fit(plans_scaled)

# Load intent responses for multi-step troubleshooting
import json
with open("intent_responses.json") as f:
    intent_responses = json.load(f)

# -------------------- Helper Functions --------------------
def extract_plan_requirements(text):
    """
    Extract user requirements for plan recommendation from text.
    Returns a dictionary with keys: offer_price, validity, data_per_day, sms_per_day
    """
    offer_price = None
    validity = None
    data_per_day = None
    sms_per_day = None

    price_match = re.search(r'(\d+)\s*(?:rs|₹)', text, re.IGNORECASE)
    if price_match:
        offer_price = float(price_match.group(1))

    validity_match = re.search(r'(\d+)\s*(?:days|day)', text, re.IGNORECASE)
    if validity_match:
        validity = float(validity_match.group(1))

    data_match = re.search(r'(\d+\.?\d*)\s*GB\s*(?:per\s*day)?', text, re.IGNORECASE)
    if data_match:
        data_per_day = float(data_match.group(1))

    sms_match = re.search(r'(\d+)\s*(?:SMS|sms)\s*(?:per\s*day)?', text, re.IGNORECASE)
    if sms_match:
        sms_per_day = float(sms_match.group(1))

    return {
        "offer_price": offer_price,
        "validity": validity,
        "data_per_day": data_per_day,
        "sms_per_day": sms_per_day
    }

def recommend_plans(user_input_text):
    user_req = extract_plan_requirements(user_input_text)
    offer_price = user_req["offer_price"] or 500
    validity = user_req["validity"] or 30
    data_per_day = user_req["data_per_day"] or 1.5
    sms_per_day = user_req["sms_per_day"] or 100
    cost_per_day = offer_price / validity

    user_input = pd.DataFrame([{
        "OFFER_PRICE": offer_price,
        "VALIDITY": validity,
        "DATA_PER_DAY": data_per_day,
        "SMS_PER_DAY": sms_per_day,
        "COST_PER_DAY": cost_per_day
    }])

    user_scaled = plans_scaler.transform(user_input)
    distances, indices = plans_knn.kneighbors(user_scaled)
    recommended_plans = plans.iloc[indices[0]].to_dict(orient='records')
    return recommended_plans

# -------------------- API Routes --------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Predict intent
    X = vectorizer.transform([text])
    predicted_intent = model.predict(X)[0]

    # -------------------- Plan Inquiry --------------------
    if predicted_intent == "plan_inquiry":
        plans_list = recommend_plans(text)
        response_text = "Here are some plans you might like based on your requirements."
        return jsonify({
            'intent': predicted_intent,
            'response': response_text,
            'plans': plans_list
        })

    # -------------------- Other intents --------------------
    response_data = intent_responses.get(predicted_intent)
    if response_data:
        return jsonify({
            "intent": predicted_intent,
            "message": response_data.get("message"),
            "steps": response_data.get("steps"),
            "quick_actions": response_data.get("quick_actions"),
            "follow_up_prompts": response_data.get("follow_up_prompts"),
            "escalate": response_data.get("escalate"),
            "raw_text": text
        })

    # Default fallback
    return jsonify({
        'intent': predicted_intent,
        'response': "Sorry, I didn't understand that. Could you rephrase?"
    })

if __name__ == "__main__":
    app.run(debug=True)

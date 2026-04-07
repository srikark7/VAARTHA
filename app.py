import os
import json
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from google import genai
from google.genai import types

class Config:
    PORT = int(os.environ.get("PORT", 3000))
    MODEL_ID = "gemini-2.5-flash" 
    API_KEY = os.environ.get("AIzaSyAf25So8PlysTYKbqAcBzUOSJbUr6tMxp8")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("VAARTHA-Backend")

client = genai.Client(api_key=Config.API_KEY)

def verify_claim(claim: str):
    prompt = f"""
    You are an elite AI fact-checker operating in the year 2026.
    
    CRITICAL SYSTEM FACTS:
    - The current year is 2026.
    - Royal Challengers Bengaluru (RCB) officially WON the men's Indian Premier League (IPL) in 2025.
    
    TASK:
    Base your answers on your internal knowledge and the system facts above.

    CLAIM TO VERIFY: "{claim}"

    Return ONLY a JSON object with:
    "verdict": (Strictly choose one: Real, Fake, or Misleading)
    "confidence": (integer between 0 and 100)
    "reasoning": (One concise sentence explaining why)
    """
    
    try:
        response = client.models.generate_content(
            model=Config.MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"AI Verification error: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/services", methods=["GET"])
def services():
    return render_template("services.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/submit_contact", methods=["POST"])
def submit_contact():
    name = request.form.get("name")
    email = request.form.get("email")
    phone = request.form.get("phone", "Not provided")
    subject = request.form.get("subject")
    message = request.form.get("message")
    
    logger.info("NEW CONTACT FORM SUBMISSION")
    logger.info(f"Name: {name}")
    logger.info(f"Email: {email}")
    logger.info(f"Phone: {phone}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Message: {message}")
    
    return jsonify({"status": "success", "message": "Message received!"})

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "").strip()
    uploaded_file = request.files.get("file")
    current_date = datetime.now().strftime("%B %d, %Y")
    
    if uploaded_file and uploaded_file.filename != "":
        if not uploaded_file.filename.endswith('.txt'):
            return render_template("result.html", prediction="Error", confidence=0, reasoning="Invalid file type. Please upload a .txt file.", analysis_date=current_date)
        try:
            news_text = uploaded_file.read().decode("utf-8").strip()
        except Exception:
            return render_template("result.html", prediction="Error", confidence=0, reasoning="Could not read the uploaded file.", analysis_date=current_date)
    
    if not news_text:
        return render_template("result.html", prediction="Error", confidence=0, reasoning="No text provided. Please enter a claim or upload a file.", analysis_date=current_date)
    
    if len(news_text) > 5000:
        news_text = news_text[:5000]
    
    logger.info(f"Analyzing Claim: {news_text[:60]}...")
    
    result = verify_claim(news_text)
    
    if not result:
        return render_template("result.html", prediction="Error", confidence=0, reasoning="Server error or API Key issue. Please try again.", analysis_date=current_date)
    
    verdict = result.get("verdict", "Unverified").capitalize()
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "Analysis complete.")
    
    return render_template("result.html", 
                           prediction=verdict, 
                           confidence=confidence, 
                           reasoning=reasoning,
                           analysis_date=current_date)

if __name__ == "__main__":
    print("VAARTHA SYSTEM FULLY OPERATIONAL")
    app.run(host="0.0.0.0", port=Config.PORT, debug=True)
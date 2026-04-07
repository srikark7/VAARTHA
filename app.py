import os
import json
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from google import genai
from google.genai import types

class Config:
    PORT = int(os.environ.get("PORT", 3000))
    # Using the latest 2.5 Flash for accuracy
    MODEL_ID = "gemini-2.5-flash"
    # Fallback model in case of heavy traffic
    LITE_MODEL = "gemini-2.5-flash-lite"
    # YOUR NEW API KEY
    API_KEY = "AIzaSyDfPAs2wWPmX0xLl3dbTdX4Kc4igBaoeHw"

app = Flask(__name__, template_folder='.')

# Logging to help you see errors in the Render/CMD console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("VAARTHA-Backend")

# Initialize Gemini Client
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
        # Try with the main 2.5 Flash model
        response = client.models.generate_content(
            model=Config.MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.warning(f"Primary model failed: {e}. Trying Lite model...")
        try:
            # Automatic fallback to Flash-Lite if 2.5 is busy
            response = client.models.generate_content(
                model=Config.LITE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e2:
            logger.error(f"Both models failed: {e2}")
            return None

# --- APP ROUTES ---

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
    logger.info(f"New contact form from: {name}")
    return jsonify({"status": "success", "message": "Message received!"})

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news", "").strip()
    uploaded_file = request.files.get("file")
    current_date = datetime.now().strftime("%B %d, %Y")
    
    if uploaded_file and uploaded_file.filename != "":
        try:
            news_text = uploaded_file.read().decode("utf-8").strip()
        except:
            pass
    
    if not news_text:
        return render_template("result.html", prediction="Error", confidence=0, reasoning="Please provide text.", analysis_date=current_date)

    result = verify_claim(news_text)
    
    if not result:
        return render_template("result.html", 
                               prediction="Service Error", 
                               confidence=0, 
                               reasoning="AI Quota exhausted. Please try again in 1-2 minutes.", 
                               analysis_date=current_date)
    
    return render_template("result.html", 
                           prediction=result.get("verdict", "Unknown").capitalize(), 
                           confidence=result.get("confidence", 0), 
                           reasoning=result.get("reasoning", "Analysis complete."),
                           analysis_date=current_date)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=True)

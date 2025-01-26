from flask import Flask, render_template, request, jsonify, Blueprint
from services.predictor import PredictionService, PredictionError, initialize_global_data
from services.polly_service import PollyService
import logging

app = Flask(__name__)
bp = Blueprint('main', __name__)

# Singleton instances
prediction_service = None
polly_service = None

@bp.before_app_request
def initialize():
    """Initialize data and models before first request"""
    global prediction_service, polly_service
    if prediction_service is None:
        # Initialize data once
        initialize_global_data()
        prediction_service = PredictionService()
        polly_service = PollyService()

def get_predictor():
    global prediction_service
    if prediction_service is None:
        # Initialize data once
        initialize_global_data()
        prediction_service = PredictionService()
    return prediction_service

def get_polly():
    global polly_service
    if polly_service is None:
        polly_service = PollyService()
    return polly_service

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        language = data.get("language")
        user_input = data.get("user_input")
        
        if not language or not user_input:
            return jsonify({"error": "Missing required fields"}), 400
            
        # Get prediction with enhanced metadata
        result = get_predictor().predict(language, user_input)
        
        # Generate audio based on language
        if language == "toronto_english":
            # Use a specific voice/style for Toronto slang
            audio_data = get_polly().synthesize_direct(
                result["predicted_text"], 
                "toronto_english"
            )
        elif language in ["us_english", "british_english"]:
            audio_data = get_polly().synthesize_direct(
                result["predicted_text"], 
                language
            )
        else:
            audio_data = get_polly().synthesize_ipa(result["predicted_text"])
            
        if audio_data:
            result["audio"] = audio_data
        
        return jsonify(result)
        
    except PredictionError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

# Register blueprint
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
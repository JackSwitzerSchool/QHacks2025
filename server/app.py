from flask import Flask, render_template, request, jsonify, Blueprint
from services.predictor import PredictionService, PredictionError, initialize_global_data
from services.polly_service import PollyService
from forecast.prediction import predict_ipa_for_time, find_nearest_vectors
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
        initialize_global_data()  # This now loads the ML models too
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
            
        # Get prediction with enhanced ML-based features
        result = prediction_service.predict(language, user_input)
        
        # Generate audio based on language and predicted IPA
        if language == "toronto_english":
            # Use Toronto-specific voice settings
            audio_data = polly_service.synthesize_direct(
                result["predicted_text"], 
                "toronto_english"
            )
        elif language in ["us_english", "british_english"]:
            # Direct synthesis for modern English
            audio_data = polly_service.synthesize_direct(
                result["predicted_text"], 
                language
            )
        else:
            # Use IPA synthesis for historical/future predictions
            audio_data = polly_service.synthesize_ipa(result["predicted_text"])
            
        if audio_data:
            result["audio"] = audio_data
        
        return jsonify(result)
        
    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Register blueprint
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
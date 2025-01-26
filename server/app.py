from flask import Flask, render_template, request, jsonify
from services.predictor import PredictionService, PredictionError

app = Flask(__name__)
predictor = PredictionService()

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
        result = predictor.predict(language, user_input)
        return jsonify(result)
        
    except PredictionError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
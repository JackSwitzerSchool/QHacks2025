from forecast.prediction import *
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    # Map frontend language selections to time periods
    LANGUAGE_TIME_MAP = {
        "PIE": -5000,  # Proto-Indo-European
        "old_english": -1100,  # Old English period
        "us_english": 0,  # Present day
        "british_english": -300,  # Present day
        "future_english_1000": 1000,  # 1000 years in future
        "future_english_2000": 2000,  # 2000 years in future
        "future_toronto": 100,  # Far future Toronto variant
    }

    def __init__(self):
        """Initialize the prediction service with required models and data"""
        try:
            # Load dataset
            self.df = load_dataset(FILE_PATH)
            self.df = parse_vectors_from_df(self.df, vector_col="vector")
            
            # Initialize BERT model
            self.tokenizer, self.model, self.device = initialize_model()
            
            logger.info("PredictionService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PredictionService: {str(e)}")
            raise PredictionError(f"Service initialization failed: {str(e)}")

    def predict(self, language: str, text: str) -> Dict[str, Any]:
        """
        Predict the phonetic representation for the given text at the specified language/time period
        
        Args:
            language: The target language/time period (key from LANGUAGE_TIME_MAP)
            text: The input text to translate
            
        Returns:
            Dict containing predicted IPA and additional metadata
        """
        try:
            # Validate language selection
            if language not in self.LANGUAGE_TIME_MAP:
                raise PredictionError(f"Unsupported language selection: {language}")
            
            # Get target time period
            target_time = self.LANGUAGE_TIME_MAP[language]
            
            # Clean input text
            text = text.strip().lower()
            if not text:
                raise PredictionError("Empty input text")
            
            # Find nearest neighbors using BERT embeddings
            neighbors_df, distances = find_nearest_vectors(
                query=text,
                df=self.df,
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.device,
                top_n=10
            )
            
            # Predict IPA for the target time period
            predicted_ipa = predict_ipa_for_time(target_time, neighbors_df)
            
            # Apply any time-specific phonological rules
            final_ipa = apply_phonological_rules(predicted_ipa, target_time)
            
            # Prepare response
            response = {
                "predicted_text": final_ipa,
                "time_period": target_time,
                "language": language,
                "original_text": text,
                "confidence_score": float(1.0 - min(distances) if len(distances) > 0 else 0.0),
                "nearest_matches": [
                    {
                        "word": row.get('word', 'N/A'),
                        "time_period": row.get('time_period', 'N/A'),
                        "ipa": row.get('phonetic_representation', 'N/A'),
                        "distance": float(distances[i])
                    }
                    for i, (_, row) in enumerate(neighbors_df.iterrows())
                ][:5]  # Include top 5 matches in response
            }
            
            logger.info(f"Successfully predicted IPA for '{text}' in {language}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}")

class PredictionError(Exception):
    pass 
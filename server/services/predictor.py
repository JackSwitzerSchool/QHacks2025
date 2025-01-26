import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forecast.prediction import *
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Module-level cache for loaded data and model
_GLOBAL_DATA = {
    'df': None,
    'tokenizer': None,
    'model': None,
    'device': None
}

def initialize_global_data():
    """Initialize data and models once at module level"""
    if _GLOBAL_DATA['df'] is None:
        logger.info("Loading dataset and initializing models...")
        try:
            # Load dataset
            _GLOBAL_DATA['df'] = load_dataset(FILE_PATH)
            _GLOBAL_DATA['df'] = parse_vectors_from_df(_GLOBAL_DATA['df'], vector_col="vector")
            
            # Initialize BERT model
            _GLOBAL_DATA['tokenizer'], _GLOBAL_DATA['model'], _GLOBAL_DATA['device'] = initialize_model()
            
            logger.info("Global data initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize global data: {str(e)}")
            raise

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
        """Initialize the prediction service using global data"""
        # Use already loaded data and models
        self.df = _GLOBAL_DATA['df']
        self.tokenizer = _GLOBAL_DATA['tokenizer']
        self.model = _GLOBAL_DATA['model']
        self.device = _GLOBAL_DATA['device']
        logger.info("PredictionService initialized using global data")

    def predict(self, language: str, text: str) -> Dict[str, Any]:
        """
        Predict the phonetic representation for the given text at the specified language/time period
        Handles both single words and full sentences.
        
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
            
            # Split into words and process each separately
            words = text.split()
            word_predictions = []
            total_confidence = 0
            all_matches = []
            
            for word in words:
                # Find nearest neighbors using BERT embeddings
                neighbors_df, distances = find_nearest_vectors(
                    query=word,
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
                
                # Store prediction and confidence
                word_confidence = float(1.0 - min(distances) if len(distances) > 0 else 0.0)
                word_predictions.append({
                    'original': word,
                    'ipa': final_ipa,
                    'confidence': word_confidence
                })
                total_confidence += word_confidence
                
                # Store nearest matches for this word
                all_matches.extend([
                    {
                        "word": row.get('word', 'N/A'),
                        "time_period": row.get('time_period', 'N/A'),
                        "ipa": row.get('phonetic_representation', 'N/A'),
                        "distance": float(distances[i]),
                        "original_word": word
                    }
                    for i, (_, row) in enumerate(neighbors_df.iterrows())
                ][:3])  # Keep top 3 matches per word
            
            # Combine predictions into sentence
            final_ipa_sentence = ' '.join(wp['ipa'] for wp in word_predictions)
            avg_confidence = total_confidence / len(words) if words else 0
            
            # Sort all matches by distance and keep top 5 overall
            all_matches.sort(key=lambda x: x['distance'])
            top_matches = all_matches[:5]
            
            # Prepare response
            response = {
                "predicted_text": final_ipa_sentence,
                "time_period": target_time,
                "language": language,
                "original_text": text,
                "confidence_score": float(avg_confidence),
                "word_predictions": word_predictions,  # Individual word predictions
                "nearest_matches": top_matches
            }
            
            logger.info(f"Successfully predicted IPA for sentence: '{text}' in {language}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}")

class PredictionError(Exception):
    pass 
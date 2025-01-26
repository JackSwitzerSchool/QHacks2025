import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forecast.prediction import (
    load_dataset, parse_vectors_from_df, initialize_model, 
    find_nearest_vectors, predict_ipa_for_time, predict_feature_change,
    interpolate_features, phone_feature_map, phone_to_bits, bits_to_phone,
    ipa_to_feature_sequence, feature_sequence_to_ipa
)
from typing import Dict, Any
import logging
from services.claude_service import ClaudeService
import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache for loaded data and model
_GLOBAL_DATA = {
    'df': None,
    'tokenizer': None,
    'model': None,
    'device': None
}

# Update FILE_PATH constant at the top
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        "datapipeline/data/output/projected_vectors_with_metadata.csv")

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
        "toronto_english": 0,  # Toronto variant
    }

    def __init__(self):
        """Initialize the prediction service using global data"""
        # Use already loaded data and models
        self.df = _GLOBAL_DATA['df']
        self.tokenizer = _GLOBAL_DATA['tokenizer']
        self.model = _GLOBAL_DATA['model']
        self.device = _GLOBAL_DATA['device']
        self.claude_service = ClaudeService()
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
            
            # Clean input text
            text = text.strip().lower()
            if not text:
                raise PredictionError("Empty input text")

            # Handle Toronto slang separately
            if language == "toronto_english":
                translation = self.claude_service.translate_to_toronto_slang(text)
                if translation is None:
                    raise PredictionError("Toronto translation failed")
                
                # Extract both text versions
                toronto_text = translation["toronto_text"]
                ipa_text = translation["ipa_text"]
                
                response = {
                    "predicted_text": toronto_text,  # Use Toronto slang for speech
                    "ipa_text": ipa_text,  # Store IPA separately
                    "time_period": self.LANGUAGE_TIME_MAP[language],
                    "language": language,
                    "original_text": text,
                    "confidence_score": 1.0,
                    "is_modern": True,
                    "is_toronto": True,
                    "explanation": toronto_text.split("\n")[0],  # First line contains explanation
                    "word_predictions": [{
                        'original': word,
                        'toronto': word,
                        'confidence': 1.0
                    } for word in toronto_text.split()],
                    "nearest_matches": []
                }
                return response

            # For modern English variants, use direct text-to-speech
            if language in ["us_english", "british_english"]:
                response = {
                    "predicted_text": text,  # Use original text instead of IPA
                    "time_period": self.LANGUAGE_TIME_MAP[language],
                    "language": language,
                    "original_text": text,
                    "confidence_score": 1.0,  # High confidence for direct synthesis
                    "is_modern": True,  # Add flag to indicate modern English
                    "word_predictions": [{
                        'original': word,
                        'ipa': word,  # No IPA needed for direct synthesis
                        'confidence': 1.0
                    } for word in text.split()],
                    "nearest_matches": []  # No historical matches needed
                }
                return response

            # For historical/future predictions
            words = text.split()
            word_predictions = []
            total_confidence = 0
            all_matches = []
            
            target_time = self.LANGUAGE_TIME_MAP[language]
            
            for word in words:
                # Get nearest neighbors
                neighbors_df, distances = find_nearest_vectors(
                    query=word,
                    df=self.df,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    device=self.device,
                    top_n=10
                )
                
                # Use appropriate prediction method based on time
                if target_time <= 0:  # Past/present
                    time_ordered = neighbors_df.sort_values('time_period')
                    # Find closest match within time window
                    time_window = 500
                    close_matches = time_ordered[
                        (time_ordered['time_period'] >= target_time - time_window) & 
                        (time_ordered['time_period'] <= target_time + time_window)
                    ]
                    
                    if not close_matches.empty:
                        # Use closest historical match if we have one in range
                        closest_idx = (close_matches['time_period'] - target_time).abs().idxmin()
                        final_ipa = close_matches.loc[closest_idx]['phonetic_representation']
                    else:
                        # No close historical match, use ML prediction
                        base_ipa = predict_ipa_for_time(target_time, neighbors_df)
                        if base_ipa:
                            features = ipa_to_feature_sequence(base_ipa)
                            time_delta = target_time - neighbors_df.iloc[0]['time_period']
                            predicted_features = [predict_feature_change(f, time_delta) for f in features]
                            final_ipa = feature_sequence_to_ipa(predicted_features)
                        else:
                            final_ipa = "/???/"
                else:  # Future
                    # Use feature-based prediction
                    base_ipa = predict_ipa_for_time(target_time, neighbors_df)
                    if base_ipa:
                        features = ipa_to_feature_sequence(base_ipa)
                        time_delta = target_time - neighbors_df.iloc[0]['time_period']
                        predicted_features = [predict_feature_change(f, time_delta) for f in features]
                        final_ipa = feature_sequence_to_ipa(predicted_features)
                    else:
                        final_ipa = "/???/"

                # Calculate confidence and store prediction
                word_confidence = float(1.0 - min(distances) if len(distances) > 0 else 0.0)
                word_predictions.append({
                    'original': word,
                    'ipa': final_ipa,
                    'confidence': word_confidence
                })
                total_confidence += word_confidence
                
                # Store matches (keep existing match storage code)
                all_matches.extend([
                    {
                        "word": row.get('word', 'N/A'),
                        "time_period": row.get('time_period', 'N/A'),
                        "ipa": row.get('phonetic_representation', 'N/A'),
                        "distance": float(distances[i]),
                        "original_word": word
                    }
                    for i, (_, row) in enumerate(neighbors_df.iterrows())
                ][:3])

            # Assemble final response (keep existing response assembly)
            final_ipa_sentence = ' '.join(wp['ipa'] for wp in word_predictions)
            avg_confidence = total_confidence / len(words) if words else 0
            
            all_matches.sort(key=lambda x: x['distance'])
            top_matches = all_matches[:5]
            
            response = {
                "predicted_text": final_ipa_sentence,
                "time_period": self.LANGUAGE_TIME_MAP[language],
                "language": language,
                "original_text": text,
                "confidence_score": float(avg_confidence),
                "word_predictions": word_predictions,
                "nearest_matches": top_matches
            }
            
            logger.info(f"Successfully predicted IPA for sentence: '{text}' in {language}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}")

class PredictionError(Exception):
    pass

def phone_to_bits(phone: str) -> np.ndarray:
    """Convert phone symbol to its 14D feature vector."""
    if phone in phone_feature_map:
        return np.array(phone_feature_map[phone], dtype=float)
    return np.zeros(14, dtype=float)  # Changed from 12 to 14 dimensions 
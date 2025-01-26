import os
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import re

# -----------------------------------------------------------------------------
# 1. SET UP LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setting environment variable to handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# 2. FILE PATHS
# -----------------------------------------------------------------------------
FILE_PATH = "datapipeline/data/output/projected_vectors_with_metadata.csv"
# Update this path to wherever your CSV file is located.

# -----------------------------------------------------------------------------
# 3. LOAD DATA & PARSE VECTORS
# -----------------------------------------------------------------------------
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from CSV and returns a Pandas DataFrame.
    The CSV must contain columns: 'word', 'time_period', 'phonetic_representation', 'vector', etc.
    """
    try:
        logger.info("Loading dataset from %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading CSV: %s", e)
        raise

def parse_vectors_from_df(df: pd.DataFrame, vector_col: str = "vector") -> pd.DataFrame:
    """
    Parses string-encoded vectors in 'vector_col' (e.g. "[0.1, 0.2, ...]") to numpy arrays.
    """
    logger.info("Parsing existing '%s' column to numpy arrays...", vector_col)

    def _parse_vector(vec_str: str):
        # Example: "[0.1,0.2,0.3]" -> array([0.1, 0.2, 0.3])
        return np.fromstring(vec_str.strip()[1:-1], sep=",")

    df[vector_col] = df[vector_col].apply(_parse_vector)
    if not df.empty:
        example_shape = df[vector_col].iloc[0].shape
        logger.info("Vector parsing complete. Example vector shape: %s", example_shape)
    else:
        logger.info("DataFrame is empty; no vectors to parse.")
    return df

# -----------------------------------------------------------------------------
# 4. MODEL INITIALIZATION
# -----------------------------------------------------------------------------
def initialize_model(model_id: str = "answerdotai/ModernBERT-base"):
    """
    Initializes and returns a tokenizer and model for the given model_id, 
    moving model to GPU if available.
    """
    try:
        logger.info("Loading tokenizer and model: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info("Model loaded successfully and moved to %s.", device)
        return tokenizer, model, device
    except Exception as e:
        logger.error("Error loading model '%s': %s", model_id, e)
        raise

# -----------------------------------------------------------------------------
# 5. NEAREST VECTOR SEARCH (With Time Period Coverage)
# -----------------------------------------------------------------------------
def find_nearest_vectors(
    query: str, 
    df: pd.DataFrame, 
    tokenizer, 
    model, 
    device: torch.device, 
    top_n: int = 10,
    vector_col: str = "vector"
):
    """
    Find nearest vectors with time period coverage.
    """
    logger.info("Generating embedding for query: '%s'", query)
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move to device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    query_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    logger.info("Computing cosine distances for the query embedding...")
    stored_vectors = np.stack(df[vector_col].values)
    distances = cdist([query_vector], stored_vectors, metric="cosine").flatten()
    
    # Get indices of top matches for each time period
    target_periods = [-5000, -1100, -300, 0]
    final_indices = []
    
    # First get best match from each target period
    temp_df = df.copy()
    temp_df['distance'] = distances
    
    for period in target_periods:
        period_matches = temp_df[temp_df['time_period'] == period]
        if not period_matches.empty:
            best_idx = period_matches['distance'].idxmin()
            final_indices.append(best_idx)
    
    # Fill remaining slots with best overall matches
    remaining = top_n - len(final_indices)
    if remaining > 0:
        temp_df = temp_df[~temp_df.index.isin(final_indices)]
        additional_indices = temp_df.nsmallest(remaining, 'distance').index
        final_indices.extend(additional_indices)
    
    # Get final results
    final_results = df.loc[final_indices].copy()
    final_distances = distances[final_indices]
    final_results['similarity'] = 1 - final_distances
    
    logger.info("Found top %d nearest vectors with time period coverage.", len(final_results))
    return final_results, final_distances

# -----------------------------------------------------------------------------
# 6. PHONOLOGICAL FEATURE INVENTORY (14-D)
# -----------------------------------------------------------------------------
phone_feature_map = {
    # [syll, cons, son, cont, voi, nas, lab, cor, dor, high, back, low, round, tense]
    
    # Vowels with more precise features
    'i':  [1,0,1,1,1,0,0,0,0,1,0,0,0,1],  # high front tense
    'ɪ':  [1,0,1,1,1,0,0,0,0,1,0,0,0,0],  # high front lax
    'e':  [1,0,1,1,1,0,0,0,0,0.7,0,0,0,1], # upper-mid front tense
    'ɛ':  [1,0,1,1,1,0,0,0,0,0.3,0,0,0,0], # mid front lax
    'æ':  [1,0,1,1,1,0,0,0,0,0,0,1,0,1],  # low front
    'ʌ':  [1,0,1,1,1,0,0,0,1,0,1,0,0,0],  # mid back unrounded
    'ɔ':  [1,0,1,1,1,0,0,0,1,0,1,0,1,1],  # mid back rounded
    'u':  [1,0,1,1,1,0,1,0,1,1,1,0,1,1],  # high back rounded
    'ə':  [1,0,1,1,1,0,0,0,0,0.5,0.5,0,0,0], # schwa
    'ɑ':  [1,0,1,1,1,0,0,0,1,0,1,1,0,0],  # low back unrounded
    'ʊ':  [1,0,1,1,1,0,1,0,1,1,1,0,1,0],  # near-high back rounded
    'o':  [1,0,1,1,1,0,0,0,1,0.5,1,0,1,1], # mid back rounded
    'a':  [1,0,1,1,1,0,0,0,1,0,0,1,0,1],  # low central
    
    # Consonants with articulatory features
    't':  [0,1,0,0,0,0,0,1,0,0,0,0,0,0],  # voiceless alveolar stop
    'd':  [0,1,0,0,1,0,0,1,0,0,0,0,0,0],  # voiced alveolar stop
    'ɾ':  [0,1,1,0,1,0,0,1,0,0,0,0,0,0],  # alveolar tap
    'ð':  [0,1,0,1,1,0,0,1,0,0,0,0,0,0],  # voiced dental fricative
    'θ':  [0,1,0,1,0,0,0,1,0,0,0,0,0,0],  # voiceless dental fricative
    'ŋ':  [0,1,1,0,1,1,0,0,1,0,0,0,0,0],  # velar nasal
    'ʃ':  [0,1,0,1,0,0,0,1,0,1,0,0,0,0],  # post-alveolar fricative
    'ʒ':  [0,1,0,1,1,0,0,1,0,1,0,0,0,0],  # voiced post-alveolar fricative
    'p':  [0,1,0,0,0,0,1,0,0,0,0,0,0,0],  # voiceless bilabial stop
    'b':  [0,1,0,0,1,0,1,0,0,0,0,0,0,0],  # voiced bilabial stop
    'f':  [0,1,0,1,0,0,1,0,0,0,0,0,0,0],  # voiceless labiodental fricative
    'v':  [0,1,0,1,1,0,1,0,0,0,0,0,0,0],  # voiced labiodental fricative
    'm':  [0,1,1,0,1,1,1,0,0,0,0,0,0,0],  # bilabial nasal
    'n':  [0,1,1,0,1,1,0,1,0,0,0,0,0,0],  # alveolar nasal
    'l':  [0,1,1,1,1,0,0,1,0,0,0,0,0,0],  # alveolar lateral
    'r':  [0,1,1,1,1,0,0,1,0,0,0,0,0,0],  # alveolar approximant
    'w':  [0,0,1,1,1,0,1,0,1,1,1,0,1,0],  # labial-velar approximant
    'j':  [0,0,1,1,1,0,0,0,1,1,0,0,0,0],  # palatal approximant
    'h':  [0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # glottal fricative
    'ʔ':  [0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # glottal stop
    'ˈ':  [0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # primary stress
    'ˌ':  [0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # secondary stress
}

def phone_to_bits(phone: str) -> np.ndarray:
    """
    Returns the 14-d feature vector for the phone,
    or a default zero vector if not present.
    """
    return np.array(phone_feature_map.get(phone, [0]*14), dtype=float)

def bits_to_phone(bits: np.ndarray) -> str:
    """
    Finds the phone in phone_feature_map with the minimum Euclidean distance
    to 'bits', returning that phone's string.
    """
    best_phone = None
    best_dist = float('inf')
    for ph, feat in phone_feature_map.items():
        dist = np.linalg.norm(bits - np.array(feat, dtype=float))
        if dist < best_dist:
            best_dist = dist
            best_phone = ph
    return best_phone

# -----------------------------------------------------------------------------
# PHONETIC SHIFT PATTERNS
# -----------------------------------------------------------------------------
VOWEL_SHIFTS = {
    'ɒ': {'target': 'ɔ', 'start': -300, 'end': 0},    # Short o raising
    'æ': {'target': 'ɛ', 'start': -1100, 'end': -300},  # Short a raising
    'aː': {'target': 'eɪ', 'start': -300, 'end': 0},   # Long a diphthongization
    'iː': {'target': 'aɪ', 'start': -300, 'end': 0},   # Long i diphthongization
    'uː': {'target': 'aʊ', 'start': -300, 'end': 0},   # Long u diphthongization
}

CONSONANT_SHIFTS = {
    'k': {'target': 'tʃ', 'start': -1100, 'end': -300, 'context': '_[ieæ]'},  # Palatalization
    'g': {'target': 'dʒ', 'start': -1100, 'end': -300, 'context': '_[ieæ]'},  # Palatalization
    'hw': {'target': 'w', 'start': -300, 'end': 0},     # Loss of /hw/
}

def ipa_to_feature_sequence(ipa: str) -> list[np.ndarray]:
    """Convert IPA string to sequence of feature vectors"""
    # Clean IPA string
    ipa = re.sub(r'[/\[\]]', '', ipa)  # Remove slashes and brackets
    
    # Split into individual phones (this is simplified, could be more sophisticated)
    phones = list(ipa)
    
    # Convert each phone to feature vector
    return [phone_to_bits(p) for p in phones]

def feature_sequence_to_ipa(features: list[np.ndarray]) -> str:
    """Convert sequence of feature vectors back to IPA string"""
    phones = [bits_to_phone(feat) for feat in features]
    return '/' + ''.join(p for p in phones if p is not None) + '/'

def predict_feature_change(feature_vector: np.ndarray, time_delta: float) -> np.ndarray:
    """
    Predicts how features change over time based on common sound change patterns
    """
    # Extract key features
    syllabic = feature_vector[0]
    consonantal = feature_vector[1]
    high = feature_vector[9]
    back = feature_vector[10]
    low = feature_vector[11]
    tense = feature_vector[13]
    
    # Scale time_delta to [0,1] range for interpolation
    alpha = min(1.0, abs(time_delta) / 2000)
    alpha = pow(alpha, 0.3)  # Non-linear change rate
    
    new_features = feature_vector.copy()
    
    if syllabic == 1:  # Vowel changes
        if time_delta > 0:  # Future changes
            # Chain shift patterns
            if high == 1 and tense == 1:  # High tense vowels tend to diphthongize
                new_features[9] -= 0.3 * alpha  # Lower height
                new_features[13] -= 0.2 * alpha  # Reduce tenseness
            elif high == 0 and low == 0:  # Mid vowels tend to raise
                new_features[9] += 0.2 * alpha  # Raise height
            elif low == 1:  # Low vowels tend to raise
                new_features[11] -= 0.3 * alpha  # Reduce lowness
                new_features[9] += 0.2 * alpha  # Increase height
        else:  # Historical changes
            # Reverse chain shifts
            if not back:  # Front vowels
                new_features[10] += 0.1 * alpha  # Slight backing
            if tense:
                new_features[13] -= 0.1 * alpha  # Reduce tenseness
    
    else:  # Consonant changes
        if time_delta > 0:
            # Lenition patterns
            if consonantal == 1:
                new_features[2] += 0.2 * alpha  # Increase sonority
                new_features[3] += 0.3 * alpha  # Increase continuancy
        else:
            # Fortition patterns
            if feature_vector[3] == 1:  # Continuants
                new_features[3] -= 0.2 * alpha  # Reduce continuancy
    
    return new_features

def interpolate_features(neighbors_df: pd.DataFrame, target_time: float) -> str:
    """
    Interpolates between attested IPA forms using feature-based transformations.
    Only used for future predictions.
    """
    # Only process future predictions
    if target_time <= 0:
        return None
        
    time_ordered = neighbors_df.sort_values('time_period')
    
    # Get most recent attested form as base
    base_ipa = time_ordered.iloc[-1]['phonetic_representation']
    time_delta = target_time - time_ordered.iloc[-1]['time_period']
    
    # Handle future extrapolation
    base_features = ipa_to_feature_sequence(base_ipa)
    predicted_features = [predict_feature_change(f, time_delta) for f in base_features]
    return feature_sequence_to_ipa(predicted_features)

def predict_ipa_for_time(time_input: float, neighbors_df: pd.DataFrame) -> str:
    """
    Predicts IPA for a given time. 
    - For past: uses nearest attested form if available, otherwise uses ML
    - For future: always uses feature-based prediction
    """
    if len(neighbors_df) == 0:
        return None

    time_ordered = neighbors_df.sort_values('time_period')
    
    # For historical times
    if time_input < 0:
        # Find closest match within a reasonable time window (±500 years)
        time_window = 500
        close_matches = time_ordered[
            (time_ordered['time_period'] >= time_input - time_window) & 
            (time_ordered['time_period'] <= time_input + time_window)
        ]
        
        if not close_matches.empty:
            # Use closest historical match if we have one in range
            closest_idx = (close_matches['time_period'] - time_input).abs().idxmin()
            return close_matches.loc[closest_idx]['phonetic_representation']
        else:
            # No close historical match, use ML prediction
            # Use the closest attested form as base
            base_idx = (time_ordered['time_period'] - time_input).abs().idxmin()
            base_ipa = time_ordered.loc[base_idx]['phonetic_representation']
            base_time = time_ordered.loc[base_idx]['time_period']
            
            # Convert to features and predict changes
            features = ipa_to_feature_sequence(base_ipa)
            time_delta = time_input - base_time
            predicted_features = [predict_feature_change(f, time_delta) for f in features]
            return feature_sequence_to_ipa(predicted_features)
    
    # For future predictions, always use feature-based interpolation
    predicted_ipa = interpolate_features(neighbors_df, time_input)
    return predicted_ipa

def apply_phonological_rules(ipa: str, time: float) -> str:
    """Apply time-dependent phonological rules"""
    result = ipa
    
    # Apply vowel shifts
    for vowel, shift in VOWEL_SHIFTS.items():
        if shift['start'] <= time <= shift['end']:
            result = result.replace(vowel, shift['target'])
    
    # Apply consonant shifts
    for cons, shift in CONSONANT_SHIFTS.items():
        if shift['start'] <= time <= shift['end']:
            if 'context' in shift:
                # Apply only in specified context
                pattern = cons + shift['context']
                replacement = shift['target'] + shift['context'][-1]
                result = re.sub(pattern, replacement, result)
            else:
                result = result.replace(cons, shift['target'])
    
    return result

# -----------------------------------------------------------------------------
# 8. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load dataset
    df = load_dataset(FILE_PATH)

    # Step 2: Parse vectors
    if "vector" in df.columns:
        df = parse_vectors_from_df(df, vector_col="vector")
    else:
        logger.error("No 'vector' column found in dataset. Exiting.")
        exit(1)

    # Step 3: Initialize model (ModernBERT)
    tokenizer, model, device = initialize_model("answerdotai/ModernBERT-base")

    # Step 4: Query inputs
    input_word = input("Enter a word to query: ").strip()
    input_time_str = input("Enter a time period (numeric) to predict IPA: ").strip()
    try:
        input_time = float(input_time_str)
    except ValueError:
        logger.error("Time must be a numeric value.")
        exit(1)

    # Step 5: Find top 10 neighbors (ensuring at least one from -5000, -1100, -300, 0 if present)
    top_neighbors_df, distances_arr = find_nearest_vectors(
        query=input_word,
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        top_n=10
    )

    # Display them
    print("\nTop 10 Neighbors:")
    for rank, (idx, row) in enumerate(top_neighbors_df.iterrows(), start=1):
        dist_idx = rank - 1
        print(f"[Rank {rank}] "
              f"Word: {row.get('word','N/A')} | "
              f"Time: {row.get('time_period','N/A')} | "
              f"IPA: {row.get('phonetic_representation','N/A')} | "
              f"Distance: {distances_arr[dist_idx]:.4f}")

    # Step 6: Predict IPA
    predicted_ipa = predict_ipa_for_time(input_time, top_neighbors_df)

    print(f"\nPredicted IPA for time = {input_time}:")
    print(f"IPA: {predicted_ipa}")
    
    # Show progression through extended timeline
    print("\nIPA progression through extended timeline:")
    historical_times = sorted(list(set([n['time_period'] for _, n in top_neighbors_df.iterrows()])))
    future_times = [0, 1000, 2000, 3000, 4000, 5000]  # Future predictions
    all_times = historical_times + future_times
    
    for t in all_times:
        pred = predict_ipa_for_time(t, top_neighbors_df)
        era = "FUTURE" if t > 0 else "PAST"
        print(f"Time {t:>6.0f} ({era}): {pred}")

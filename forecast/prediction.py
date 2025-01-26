import os
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. SET UP LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setting environment variable to handle OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------------------------------------------
# 2. FILE PATHS
# -----------------------------------------------------------------------------
FILE_PATH = "datapipeline/data/output/projected_vectors_with_metadata.csv"  # CSV with precomputed vectors

# -----------------------------------------------------------------------------
# 3. LOAD DATA & PARSE VECTORS
# -----------------------------------------------------------------------------
def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the dataset from CSV and returns a Pandas DataFrame."""
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
    Parses the string-encoded vectors in 'vector_col' to numpy arrays.
    """
    logger.info("Parsing existing '%s' column to numpy arrays...", vector_col)

    def _parse_vector(vec_str: str):
        # Removes the brackets [ and ] before parsing
        return np.fromstring(vec_str.strip()[1:-1], sep=",")

    df[vector_col] = df[vector_col].apply(_parse_vector)
    logger.info("Vector parsing complete. Example vector shape: %s", df[vector_col].iloc[0].shape)
    return df

# -----------------------------------------------------------------------------
# 4. MODEL INITIALIZATION
# -----------------------------------------------------------------------------
def initialize_model(model_id: str = "answerdotai/ModernBERT-base"):
    """
    Initializes and returns a tokenizer and model for the given model_id.
    Moves model to GPU if available.
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
# 5. NEAREST VECTOR SEARCH
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
    Given a query string, finds nearest vectors ensuring representation from different time periods
    (t=-1100, t=-5000, t=-300, t=0). Returns top matches with at least one from each period if possible,
    then fills remaining slots with overall closest matches.
    """
    logger.info("Generating embedding for query: '%s'", query)
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    query_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    logger.info("Computing cosine distances for the query embedding...")
    stored_vectors = np.stack(df[vector_col].values)  # shape: (num_rows, hidden_dim)

    # Get more candidates than needed to ensure matches from each period
    initial_k = min(100, len(df))
    distances = cdist([query_vector], stored_vectors, metric="cosine").flatten()
    
    # Create results DataFrame with distances
    results_df = df.copy()
    results_df['similarity'] = 1 - distances  # Convert distance to similarity
    results_df = results_df.nlargest(initial_k, 'similarity')
    
    # Target time periods
    target_periods = [-5000, -1100, -300, 0]
    
    # Initialize final results
    final_indices = []
    
    # First, get the closest match for each target period
    for period in target_periods:
        period_matches = results_df[results_df['time_period'] == period]
        if not period_matches.empty:
            best_match_idx = period_matches.index[0]
            final_indices.append(best_match_idx)
            results_df = results_df[~results_df.index.isin([best_match_idx])]
    
    # Fill remaining slots with best remaining matches
    remaining_slots = top_n - len(final_indices)
    if remaining_slots > 0:
        remaining_best = results_df.nlargest(remaining_slots, 'similarity')
        final_indices.extend(remaining_best.index)
    
    # Get final results and their distances
    final_results = df.loc[final_indices]
    final_distances = distances[final_indices]
    
    logger.info("Found top %d nearest vectors with time period representation.", top_n)
    return final_results, final_distances

# -----------------------------------------------------------------------------
# 6. TIME-BASED IPA PREDICTION (USING TOP 10 NEIGHBORS)
# -----------------------------------------------------------------------------
def predict_ipa_from_time(time_input: float, top_neighbors: pd.DataFrame) -> str:
    """
    Given a time_input (float) and a DataFrame 'top_neighbors' with 'time_period' 
    and 'phonetic_representation', returns a predicted IPA by time-weighted voting 
    among these neighbors.
    """
    if len(top_neighbors) == 0:
        logger.warning("No neighbors provided; cannot predict IPA.")
        return None

    # We'll accumulate weights by IPA
    ipa_weights = defaultdict(float)

    for _, row in top_neighbors.iterrows():
        neighbor_time = row["time_period"]
        neighbor_ipa = row["phonetic_representation"]  # or row['ipa'] if different column
        if pd.isna(neighbor_time) or not isinstance(neighbor_time, (int, float)):
            # Skip any invalid or missing time_period
            continue

        time_diff = abs(time_input - neighbor_time)
        # Weight: inverse of time difference +1 to avoid division by zero
        weight = 1 / (time_diff + 1)
        ipa_weights[neighbor_ipa] += weight

    if not ipa_weights:
        logger.warning("No valid time_period among neighbors; can't predict.")
        return None

    # Find the IPA with the maximum total weight
    best_ipa = max(ipa_weights, key=ipa_weights.get)
    return best_ipa

# -----------------------------------------------------------------------------
# 7. MAIN EXECUTION LOGIC (DEMO)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load dataset (with precomputed vectors)
    df = load_dataset(FILE_PATH)

    # Step 2: Parse string-encoded vectors into numpy arrays
    if "vector" in df.columns:
        df = parse_vectors_from_df(df, vector_col="vector")
    else:
        logger.error("No 'vector' column found. Please generate or provide vectors first.")
        exit(1)

    # Step 3: Initialize model/tokenizer/device for queries
    tokenizer, model, device = initialize_model("answerdotai/ModernBERT-base")

    # Step 4: Prompt user for input
    input_word = input("Enter a word to query: ").strip()
    input_time_str = input("Enter a time period (numeric) to predict IPA: ").strip()

    # Convert time input to float (or int) if valid
    try:
        input_time = float(input_time_str)
    except ValueError:
        logger.error("Invalid time input. Must be a numeric value.")
        exit(1)

    # Step 5: Find the 10 nearest neighbors for the input word
    top_n = 10
    nearest_df, nearest_dists = find_nearest_vectors(
        query=input_word, 
        df=df, 
        tokenizer=tokenizer, 
        model=model, 
        device=device, 
        top_n=top_n,
        vector_col="vector"
    )

    # Optional: print them out for inspection
    print("\nTop 10 Neighbors:")
    for rank, (idx, row) in enumerate(nearest_df.iterrows(), start=1):
        print(f"[Rank {rank}] Word: {row.get('word','N/A')} | Time: {row.get('time_period','N/A')} "
              f"| IPA: {row.get('phonetic_representation','N/A')} | Distance: {nearest_dists[rank-1]:.4f}")

    # Step 6: Predict the IPA for the given time, using the top 10 neighbors
    predicted_ipa = predict_ipa_from_time(input_time, nearest_df)
    print("\nPredicted IPA for time =", input_time, "is:", predicted_ipa)

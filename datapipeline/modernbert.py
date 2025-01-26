import os
import logging
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import signal
import psutil
from pathlib import Path

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
FILE_PATH = "C:/Users/jacks/Documents/Life/Projects/Current/Chorono/datapipeline/data/input/combined_language_data_with_translation_cleaned.csv"
OUTPUT_FILE = "C:/Users/jacks/Documents/Life/Projects/Current/Chorono/datapipeline/data/output/projected_vectors_with_metadata.csv"

# -----------------------------------------------------------------------------
# 3. LOAD DATA
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

# -----------------------------------------------------------------------------
# 4. MODEL INITIALIZATION
# -----------------------------------------------------------------------------
def initialize_model(model_id: str = "answerdotai/ModernBERT-base"):
    """
    Initializes and returns a tokenizer and model for the given model_id.
    Defaults to 'answerdotai/ModernBERT-base'.
    """
    try:
        logger.info("Loading tokenizer and model: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()
        logger.info("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error("Error loading model '%s': %s", model_id, e)
        raise

# -----------------------------------------------------------------------------
# 5. BATCH EMBEDDING GENERATION
# -----------------------------------------------------------------------------
def generate_vectors_in_batches(
    texts: list[str], 
    tokenizer, 
    model, 
    batch_size: int = 32
) -> np.ndarray:
    """
    Generates embeddings for a list of texts in batches using the provided tokenizer and model.
    Returns a numpy array of shape (num_texts, hidden_dim).
    """
    logger.info("Generating vectors in batches of size %d...", batch_size)
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # CLS token embeddings for each sentence in the batch
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_vectors.append(cls_vectors)

        logger.debug("Processed batch %d/%d", (i // batch_size) + 1, 
                     (len(texts) + batch_size - 1) // batch_size)
    
    # Concatenate all batch results into a single array
    all_vectors = np.concatenate(all_vectors, axis=0)
    logger.info("Finished generating vectors. Shape: %s", all_vectors.shape)
    return all_vectors

# -----------------------------------------------------------------------------
# 6. VECTOR STORAGE / PARSING
# -----------------------------------------------------------------------------
def store_vectors_in_df(df: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    """
    Stores a matrix of vectors into the 'vector' column of the dataframe as strings.
    """
    logger.info("Storing vectors in the DataFrame...")
    # Ensure vector dimension matches df length
    if len(vectors) != len(df):
        logger.error("Mismatch between number of vectors (%d) and DataFrame rows (%d).", 
                     len(vectors), len(df))
        raise ValueError("Number of vectors does not match the DataFrame rows.")

    df["vector"] = [np.array2string(v, separator=",") for v in vectors]
    logger.info("Vectors stored as strings in 'vector' column.")
    return df

def parse_vectors_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the string-encoded vectors in 'vector' column to numpy arrays.
    """
    logger.info("Parsing existing 'vector' column to numpy arrays...")
    
    def _parse_vector(vec_str: str):
        # Removes the brackets [ and ] before parsing
        return np.fromstring(vec_str.strip()[1:-1], sep=",")

    df["vector"] = df["vector"].apply(_parse_vector)
    logger.info("Vector parsing complete.")
    return df

# -----------------------------------------------------------------------------
# 7. NEAREST VECTOR SEARCH
# -----------------------------------------------------------------------------
def find_nearest_vectors(query: str, df: pd.DataFrame, tokenizer, model, top_n: int = 5):
    """
    Given a query string, computes the embedding using the same tokenizer/model pipeline,
    compares it to all stored embeddings in df['vector'], and returns top_n matches.
    """
    logger.info("Generating embedding for query: '%s'", query)
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    logger.info("Computing cosine distances for the query embedding...")
    stored_vectors = np.stack(df["vector"].values)  # shape: (num_rows, hidden_dim)

    # Using cosine distance => smaller distance = more similar
    distances = cdist([query_vector], stored_vectors, metric="cosine").flatten()
    
    # Get the indices of the top_n nearest vectors
    nearest_indices = np.argsort(distances)[:top_n]
    nearest_results = df.iloc[nearest_indices]
    nearest_distances = distances[nearest_indices]

    logger.info("Found top %d nearest vectors.", top_n)
    return nearest_results, nearest_distances

def query_word_context(query_word: str, df: pd.DataFrame, tokenizer, model, top_n: int = 5):
    """
    Wrapper function to find and display nearest context vectors for a given query_word.
    """
    logger.info("Querying for word: '%s'", query_word)
    nearest_results, nearest_distances = find_nearest_vectors(
        query=query_word, 
        df=df, 
        tokenizer=tokenizer, 
        model=model, 
        top_n=top_n
    )

    # Display results
    print("\nNearest Context Vectors:")
    for i, (index, row) in enumerate(nearest_results.iterrows()):
        print(f"\nRank {i+1}:")
        print(f"Word: {row.get('word', 'N/A')}")
        print(f"English Translation: {row.get('english_translation', 'N/A')}")
        print(f"Language: {row.get('language', 'N/A')}")
        print(f"IPA (Phonetic Representation): {row.get('phonetic_representation', 'N/A')}")
        print(f"Time Period: {row.get('time_period', 'N/A')}")
        print(f"Distance: {nearest_distances[i]}")

    return nearest_results

# -----------------------------------------------------------------------------
# 8. MAIN EXECUTION LOGIC
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Load dataset
    df = load_dataset(FILE_PATH)

    # Step 2: Check if 'vector' column exists; if not, generate vectors in batches
    if "vector" not in df.columns:
        logger.info("No 'vector' column found. Generating vectors...")

        # Initialize model/tokenizer
        tokenizer, model = initialize_model("answerdotai/ModernBERT-base")

        # Generate vectors for the 'english_translation' column
        texts = df["english_translation"].astype(str).tolist()
        vectors = generate_vectors_in_batches(texts, tokenizer, model, batch_size=64)

        # Store vectors in DataFrame
        df = store_vectors_in_df(df, vectors)

        # Save the updated DataFrame
        logger.info("Saving dataset with vectors to %s", OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        logger.info("Parsing existing 'vector' column...")
        df = parse_vectors_from_df(df)
        # Initialize model/tokenizer for querying
        tokenizer, model = initialize_model("answerdotai/ModernBERT-base")

    # Example query
    query_word = "believe"
    top_n = 5
    results = query_word_context(query_word, df, tokenizer, model, top_n)

class ModernBERTProcessor:
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", 
                 max_workers: int = 4,
                 checkpoint_dir: str = "datapipeline/data/checkpoints"):
        """Initialize ModernBERT processor with batch processing capabilities"""
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup interrupt handling
        signal.signal(signal.SIGINT, self.handle_interrupt)
        self.interrupted = False
        
        # Initialize model and tokenizer
        logger.info(f"Loading {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Initialize tracking
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.last_checkpoint = 0
        self.checkpoint_interval = 1000

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal gracefully"""
        print("\nInterrupt received. Completing current batch and saving checkpoint...")
        self.interrupted = True

    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {str(e)}")
            raise

    def process_batch(self, texts: list, batch_size: int = 32) -> list:
        """Process a batch of texts and return their embeddings"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            if self.interrupted:
                break
                
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
                
        return embeddings

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """Process entire dataset with progress tracking and checkpointing"""
        self.start_time = time.time()
        texts = df["english_translation"].astype(str).tolist()
        total_texts = len(texts)
        processed_data = []
        
        logger.info(f"Processing {total_texts} texts in batches of {batch_size}")
        
        with tqdm(total=total_texts, desc="Processing texts") as pbar:
            for i in range(0, total_texts, batch_size):
                if self.interrupted:
                    break
                    
                batch_texts = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.process_batch(batch_texts, batch_size)
                    processed_data.extend(batch_embeddings)
                    
                    # Update progress
                    current_count = len(processed_data)
                    pbar.update(len(batch_embeddings))
                    
                    # Calculate statistics
                    elapsed_time = time.time() - self.start_time
                    texts_per_second = current_count / elapsed_time if elapsed_time > 0 else 0
                    memory_usage = self.get_memory_usage()
                    
                    pbar.set_postfix({
                        'Texts/s': f"{texts_per_second:.2f}",
                        'Memory': f"{memory_usage:.1f}GB"
                    })
                    
                    # Save checkpoint if needed
                    if current_count - self.last_checkpoint >= self.checkpoint_interval:
                        self.save_checkpoint(processed_data, df.iloc[:len(processed_data)])
                        self.last_checkpoint = current_count
                        
                except Exception as e:
                    logger.error(f"Error processing batch at index {i}: {str(e)}")
                    self.error_count += len(batch_texts)
                    continue
        
        # Store vectors in DataFrame
        df['vector'] = [np.array2string(v, separator=",") for v in processed_data]
        
        return df

    def save_checkpoint(self, embeddings: list, metadata_df: pd.DataFrame):
        """Save processing checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{len(embeddings)}.npz"
        np.savez(checkpoint_path,
                embeddings=np.array(embeddings),
                metadata=metadata_df.to_dict('records'),
                timestamp=datetime.now().isoformat())
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def find_nearest_vectors(self, query: str, df: pd.DataFrame, top_n: int = 5):
        """Find nearest vectors to query (maintaining original functionality)"""
        query_embedding = self.generate_embedding(query)
        
        # Parse stored vectors
        stored_vectors = np.stack([
            np.fromstring(vec_str.strip()[1:-1], sep=",") 
            for vec_str in df["vector"].values
        ])
        
        # Compute distances
        distances = cdist([query_embedding], stored_vectors, metric="cosine").flatten()
        nearest_indices = np.argsort(distances)[:top_n]
        
        return df.iloc[nearest_indices], distances[nearest_indices]

# Example usage
if __name__ == "__main__":
    processor = ModernBERTProcessor()
    
    # Load dataset
    df = pd.read_csv(FILE_PATH)
    
    # Process dataset if vectors don't exist
    if "vector" not in df.columns:
        df = processor.process_dataset(df, batch_size=64)
        df.to_csv(OUTPUT_FILE, index=False)
    
    # Example query
    query_word = "believe"
    results, distances = processor.find_nearest_vectors(query_word, df)
    
    # Display results
    print("\nNearest Context Vectors:")
    for i, (_, row) in enumerate(results.iterrows()):
        print(f"\nRank {i+1}:")
        print(f"Word: {row.get('word', 'N/A')}")
        print(f"English Translation: {row.get('english_translation', 'N/A')}")
        print(f"Language: {row.get('language', 'N/A')}")
        print(f"IPA: {row.get('phonetic_representation', 'N/A')}")
        print(f"Time Period: {row.get('time_period', 'N/A')}")
        print(f"Distance: {distances[i]}")


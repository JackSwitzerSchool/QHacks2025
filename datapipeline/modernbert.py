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
from torch.cuda.amp import autocast
import faiss

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
    batch_size: int = 512
) -> np.ndarray:
    """
    Generates embeddings for a list of texts in batches using the provided tokenizer and model.
    Optimized for GPU performance.
    """
    logger.info("Generating vectors in batches of size %d...", batch_size)
    all_vectors = []
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).half()  # Move to GPU and convert to fp16
    logger.info(f"Using device: {device}")
    
    # Calculate total number of batches for progress bar
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Create progress bar
    with tqdm(total=len(texts), desc="Generating vectors", unit="texts") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Move inputs to GPU
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate embeddings with mixed precision
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs)
                cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_vectors.append(cls_vectors)
            
            # Update progress bar
            pbar.update(len(batch_texts))
            pbar.set_postfix({
                'Batch': f"{(i//batch_size)+1}/{total_batches}",
                'GPU Mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            })
            
            # Optional: Clear GPU cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # Concatenate all batch results into a single array
    all_vectors = np.concatenate(all_vectors, axis=0)
    logger.info("Finished generating vectors. Shape: %s", all_vectors.shape)
    return all_vectors

# -----------------------------------------------------------------------------
# 6. VECTOR STORAGE / PARSING
# -----------------------------------------------------------------------------
def store_vectors_in_df(df: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    """Store vectors directly in DataFrame as strings"""
    logger.info("Storing vectors in DataFrame...")
    
    # Convert vectors to strings with fixed precision to save space
    df["vector"] = [np.array2string(v, separator=',', precision=8, suppress_small=True) for v in vectors]
    
    logger.info("Vectors stored in DataFrame")
    return df

def load_vectors(vector_file: str) -> np.ndarray:
    """Load vectors efficiently from npz file"""
    logger.info(f"Loading vectors from {vector_file}")
    return np.load(vector_file)["vectors"]

def parse_vectors_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse vectors using parallel processing"""
    logger.info("Parsing vectors in parallel...")
    
    def parse_vector(vec_str: str):
        return np.fromstring(vec_str.strip()[1:-1], sep=",")
    
    with ThreadPoolExecutor() as executor:
        df["vector"] = list(executor.map(parse_vector, df["vector"]))
    
    return df

# -----------------------------------------------------------------------------
# 7. NEAREST VECTOR SEARCH
# -----------------------------------------------------------------------------
def find_nearest_vectors(query: str, df: pd.DataFrame, tokenizer, model, top_n: int = 5):
    """Find nearest vectors using cosine similarity"""
    logger.info("Generating embedding for query: '%s'", query)
    
    # Move model to GPU if not already there
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Generate query embedding
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        query_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

    # Parse stored vectors
    logger.info("Parsing stored vectors...")
    stored_vectors = np.stack([
        np.fromstring(vec_str.strip()[1:-1], sep=",") 
        for vec_str in df["vector"].values
    ])

    # Compute distances
    logger.info("Computing cosine distances...")
    distances = cdist([query_vector], stored_vectors, metric="cosine").flatten()
    
    nearest_indices = np.argsort(distances)[:top_n]
    return df.iloc[nearest_indices], distances[nearest_indices]

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

    # Step 2: Check if vectors exist
    if "vector" not in df.columns:
        logger.info("No vectors found. Generating vectors...")

        # Initialize model/tokenizer
        tokenizer, model = initialize_model("answerdotai/ModernBERT-base")

        # Generate vectors
        texts = df["english_translation"].astype(str).tolist()
        vectors = generate_vectors_in_batches(texts, tokenizer, model, batch_size=512)

        # Store vectors in DataFrame
        df = store_vectors_in_df(df, vectors)

        # Save the DataFrame
        logger.info("Saving dataset with vectors to %s", OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, index=False)

    # Example queries
    test_queries = ["believe", "trust", "faith"]
    for query in test_queries:
        logger.info(f"\nQuerying for: {query}")
        results, distances = find_nearest_vectors(query, df, tokenizer, model, top_n=5)
        
        print(f"\nNearest matches for '{query}':")
        for i, (_, row) in enumerate(results.iterrows()):
            print(f"\nRank {i+1}:")
            print(f"Word: {row.get('word', 'N/A')}")
            print(f"English Translation: {row.get('english_translation', 'N/A')}")
            print(f"Language: {row.get('language', 'N/A')}")
            print(f"IPA: {row.get('phonetic_representation', 'N/A')}")
            print(f"Time Period: {row.get('time_period', 'N/A')}")
            print(f"Distance: {distances[i]:.4f}")

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
        
        # Enable mixed precision
        self.model.half()  # Convert model to fp16
        logger.info("Enabled mixed precision (fp16)")

        self.index = None

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

    def process_batch(self, texts: list, batch_size: int = 512) -> list:
        """Process a batch of texts with mixed precision"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            if self.interrupted:
                break
                
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use autocast for mixed precision
            with torch.no_grad(), torch.amp.autocast():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
                
        return embeddings

    def process_dataset(self, df: pd.DataFrame, batch_size: int = 512) -> pd.DataFrame:
        """Process dataset with optimized preprocessing"""
        # Preprocess all texts at once
        texts = df["english_translation"].astype(str).tolist()
        
        # Pre-tokenize all texts
        logger.info("Pre-tokenizing texts...")
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Process in batches
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with tqdm(total=len(texts), desc="Processing", unit="texts") as pbar:
            for i in range(0, len(texts), batch_size):
                if self.interrupted:
                    break
                    
                batch_dict = {
                    k: v[i:i + batch_size].to(self.device) 
                    for k, v in tokenized.items()
                }
                
                with torch.no_grad(), autocast():
                    outputs = self.model(**batch_dict)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(embeddings)
                
                pbar.update(len(embeddings))
                pbar.set_postfix({
                    'Batch': f"{(i//batch_size)+1}/{total_batches}",
                    'Memory': f"{self.get_memory_usage():.1f}GB"
                })
        
        vectors = np.concatenate(all_embeddings)
        df['vector'] = [np.array2string(v, separator=",") for v in vectors]
        
        return df

    def save_checkpoint(self, embeddings: list, metadata_df: pd.DataFrame):
        """Save processing checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{len(embeddings)}.npz"
        np.savez(checkpoint_path,
                embeddings=np.array(embeddings),
                metadata=metadata_df.to_dict('records'),
                timestamp=datetime.now().isoformat())
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def build_search_index(self, vectors: np.ndarray):
        """Build FAISS index for fast similarity search"""
        logger.info("Building FAISS index for fast search...")
        dimension = vectors.shape[1]
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)  # In-place L2 normalization
        
        # Use IndexFlatL2 with normalized vectors (equivalent to cosine similarity)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors.astype('float32'))
    
    def find_nearest_vectors(self, query: str, df: pd.DataFrame, top_n: int = 5):
        """Find nearest vectors using FAISS"""
        query_embedding = self.generate_embedding(query)
        
        # Build index if not exists
        if self.index is None:
            stored_vectors = np.stack([
                np.fromstring(vec_str.strip()[1:-1], sep=",") 
                for vec_str in df["vector"].values
            ])
            self.build_search_index(stored_vectors)
        
        # Normalize query vector
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search using FAISS
        distances, indices = self.index.search(query_embedding, top_n)
        
        # Convert L2 distances to similarities (smaller distance = more similar)
        similarities = 1 - distances/2  # Convert L2 distance to cosine similarity
        
        return df.iloc[indices[0]], similarities[0]

# Example usage
if __name__ == "__main__":
    processor = ModernBERTProcessor()
    
    # Load dataset
    df = pd.read_csv(FILE_PATH)
    
    # Process dataset if vectors don't exist
    if "vector" not in df.columns:
        df = processor.process_dataset(df, batch_size=512)
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
        print(f"Distance: {distances}")


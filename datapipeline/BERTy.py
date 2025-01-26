from transformers import AutoModel, AutoTokenizer
import epitran
import pandas as pd
from typing import Dict, List, Union
import logging
import numpy as np
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import signal
import psutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTy:
    def __init__(self, model_name: str = "xlm-roberta-base", max_workers: int = 4, 
                 checkpoint_dir: str = "datapipeline/data/checkpoints"):
        """Initialize the BERTy embedder with XLM-RoBERTa model"""
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup interrupt handling
        signal.signal(signal.SIGINT, self.handle_interrupt)
        self.interrupted = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        logger.info("Model loaded successfully")
        
        # Initialize Epitran with error handling
        try:
            self.epi = epitran.Epitran('eng-Latn')  # Default to English
        except UnicodeDecodeError:
            logger.warning("Failed to initialize Epitran with default encoding. Using fallback method.")
            self.epi = None
        
        # Define supported languages
        self.supported_languages = {
            'Proto-Indo-European': 'pie',
            'Old-English': 'ang',
            'English-US': 'eng-US',
            'English-UK': 'eng-GB',
            'Latin': 'lat'
        }

        # Initialize tracking
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.last_checkpoint = 0
        self.checkpoint_interval = 1000  # Save every 1000 words

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signal gracefully"""
        print("\nInterrupt received. Completing current batch and saving checkpoint...")
        self.interrupted = True
        
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return memory_gb
        
    def save_checkpoint(self, processed_data, current_index, total_words):
        """Save checkpoint with progress information"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{current_index}.npz"
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        # Save to temporary file first
        np.savez(temp_path,
                embeddings=np.array([d['embedding'] for d in processed_data]),
                metadata=[d['metadata'] for d in processed_data],
                progress={'current_index': current_index, 
                         'total_words': total_words,
                         'timestamp': datetime.now().isoformat()})
        
        # Rename to final checkpoint file
        temp_path.rename(checkpoint_path)
        
        # Remove older checkpoints, keep last 3
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.npz'))
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()
            
        memory_usage = self.get_memory_usage()
        logger.info(f"\nCheckpoint saved at index {current_index}/{total_words}")
        logger.info(f"Memory usage: {memory_usage:.2f} GB")
        
        return checkpoint_path
        
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint if it exists"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.npz'))
            if not checkpoints:
                return None, 0
            
            latest = checkpoints[-1]
            logger.info(f"Loading checkpoint: {latest}")
            
            checkpoint = np.load(latest, allow_pickle=True)
            
            # Handle both old and new checkpoint formats
            if 'progress' in checkpoint:
                progress = checkpoint['progress'].item()
                current_index = progress['current_index']
            else:
                # If using old format or corrupt checkpoint, start fresh
                logger.warning("Checkpoint format not recognized. Starting fresh.")
                return None, 0
            
            embeddings = checkpoint['embeddings']
            metadata = checkpoint['metadata']
            
            return (list(zip(embeddings, metadata)), current_index)
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {str(e)}. Starting fresh.")
            return None, 0

    def generate_embedding(self, word: str) -> np.ndarray:
        """Generate word embedding using mDeBERTa"""
        try:
            # Tokenize and convert to tensor
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling instead of just CLS token for better representation
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for word '{word}': {str(e)}")
            raise

    def get_ipa_transcription(self, word: str, language: str) -> str:
        """Get IPA transcription for a word based on language"""
        try:
            # Special handling for historical languages
            if language in ['Proto-Indo-European', 'Old-English', 'English-US', 'English-UK', 'Latin']:
                return word  # Return original form as these often come with their own transcription
            
            # For modern English variants
            if self.epi is None:
                logger.warning(f"Epitran unavailable. Returning original word: {word}")
                return word
            
            return self.epi.transliterate(word)
            
        except Exception as e:
            logger.warning(f"Could not generate IPA for '{word}' in {language}: {str(e)}")
            return word

    def create_metadata(self, word_data: Dict) -> Dict:
        """Create metadata dictionary for a word"""
        try:
            metadata = {
                'word': word_data['word'],
                'language': word_data.get('language', 'Unknown'),
                'translation': word_data['english_translation'],
                'phonetic': word_data.get('phonetic_representation', ''),
                'time_period': word_data.get('time_period', 'Unknown'),
            }
            return metadata
            
        except KeyError as e:
            logger.error(f"Missing required field in word_data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")
            raise

    def process_single_word(self, word_data: Dict) -> Dict:
        """Process a single word with error handling"""
        try:
            embedding = self.generate_embedding(word_data['english_translation'])
            metadata = self.create_metadata(word_data)
            self.processed_count += 1
            return {'embedding': embedding, 'metadata': metadata, 'success': True}
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing word '{word_data.get('word', 'unknown')}': {str(e)}")
            return {'success': False, 'error': str(e), 'word': word_data.get('word', 'unknown')}

    def process_word_batch(self, words_data: List[Dict]) -> List[Dict]:
        """Process a batch of words using parallel processing with checkpointing"""
        processed_data = []
        failed_words = []
        self.start_time = time.time()
        
        try:
            # Try to load from checkpoint
            checkpoint_data, start_index = self.load_latest_checkpoint()
            if checkpoint_data:
                processed_data = [{'embedding': e, 'metadata': m, 'success': True} 
                                for e, m in checkpoint_data]
                logger.info(f"Resumed from checkpoint with {len(processed_data)} processed words")
                words_data = words_data[start_index:]
            else:
                start_index = 0
                logger.info("Starting fresh processing run")
            
            # Create progress bar
            pbar = tqdm(total=len(words_data), initial=start_index,
                       desc="Processing words", unit="words", dynamic_ncols=True)
            
            def update_progress(future):
                if self.interrupted:
                    return
                    
                result = future.result()
                if result['success']:
                    processed_data.append(result)
                else:
                    failed_words.append(result)
                
                current_index = start_index + len(processed_data) + len(failed_words)
                pbar.update(1)
                
                # Calculate and display statistics
                elapsed_time = time.time() - self.start_time
                words_per_second = current_index / elapsed_time if elapsed_time > 0 else 0
                memory_usage = self.get_memory_usage()
                
                pbar.set_postfix({
                    'Success': f"{len(processed_data)}/{len(words_data) + start_index}", 
                    'Errors': len(failed_words),
                    'Words/s': f"{words_per_second:.2f}",
                    'Memory': f"{memory_usage:.1f}GB"
                })
                
                # Save checkpoint if needed
                if (len(processed_data) % self.checkpoint_interval == 0 and 
                    len(processed_data) > self.last_checkpoint):
                    self.save_checkpoint(processed_data, current_index, 
                                      len(words_data) + start_index)
                    self.last_checkpoint = len(processed_data)

            # Process words in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for word_data in words_data:
                    if self.interrupted:
                        break
                    future = executor.submit(self.process_single_word, word_data)
                    future.add_done_callback(update_progress)
                    futures.append(future)
                
                # Wait for all futures to complete
                for future in as_completed(futures):
                    if self.interrupted:
                        break
                    pass

            pbar.close()
            
            # Save final checkpoint if interrupted
            if self.interrupted:
                current_index = start_index + len(processed_data) + len(failed_words)
                self.save_checkpoint(processed_data, current_index, 
                                  len(words_data) + start_index)
                logger.info("Processing interrupted and progress saved")
                
            # Log final statistics
            self.log_final_statistics(processed_data, failed_words, start_index)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def log_final_statistics(self, processed_data, failed_words, start_index):
        """Log final processing statistics"""
        elapsed_time = time.time() - self.start_time
        memory_usage = self.get_memory_usage()
        
        logger.info(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully processed: {len(processed_data)} words")
        logger.info(f"Failed to process: {len(failed_words)} words")
        logger.info(f"Average processing speed: {len(processed_data)/elapsed_time:.2f} words/second")
        logger.info(f"Final memory usage: {memory_usage:.2f} GB")
        
        if failed_words:
            logger.warning("\nFailed words summary:")
            for fail in failed_words[:10]:
                logger.warning(f"Word: {fail['word']}, Error: {fail['error']}")
            if len(failed_words) > 10:
                logger.warning(f"... and {len(failed_words) - 10} more failures")

    def process_csv_file(self, file_path: str) -> pd.DataFrame:
        """Process words from a CSV file and return enriched DataFrame"""
        try:
            df = pd.read_csv(file_path)
            total_rows = len(df)
            logger.info(f"Processing {total_rows} entries...")
            
            # Remove any rows with empty translations
            df = df.dropna(subset=['english_translation'])
            cleaned_rows = len(df)
            if cleaned_rows < total_rows:
                logger.warning(f"Removed {total_rows - cleaned_rows} rows with empty translations")
            logger.info(f"Processing {cleaned_rows} entries after cleaning")
            
            # Create a list to store embeddings
            embeddings = []
            
            # Process each row
            logger.info("Generating embeddings...")
            with tqdm(total=len(df), desc="Processing words") as pbar:
                for _, row in df.iterrows():
                    try:
                        embedding = self.generate_embedding(row['english_translation'])
                        embeddings.append(embedding)
                        self.processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing word '{row.get('word', 'unknown')}': {str(e)}")
                        embeddings.append(None)
                        self.error_count += 1
                    pbar.update(1)
            
            # Add embeddings as a new column
            df['vector'] = embeddings
            
            # Save the enriched DataFrame
            output_path = Path(file_path).parent / "enriched_language_data.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved enriched data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            raise

    def save_results(self, processed_data: List[Dict], output_dir: str = "datapipeline/data/output"):
        """Save results in chunks to avoid memory issues"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: print sample metadata
        if processed_data:
            logger.info(f"Sample metadata structure: {processed_data[0]['metadata']}")
        
        # Save in chunks of 10,000 words
        chunk_size = 10000
        total_chunks = (len(processed_data) + chunk_size - 1) // chunk_size
        
        logger.info(f"\nSaving results in {total_chunks} chunks...")
        
        for i in range(0, len(processed_data), chunk_size):
            chunk = processed_data[i:i+chunk_size]
            chunk_number = i // chunk_size + 1
            
            # Create chunk filename
            chunk_file = output_dir / f"embeddings_chunk_{chunk_number}_of_{total_chunks}.npz"
            
            # Get embeddings and metadata for this chunk
            embeddings = np.array([data['embedding'] for data in chunk])
            metadata = [data['metadata'] for data in chunk]
            
            # Save chunk with timestamp
            np.savez(chunk_file,
                    embeddings=embeddings,
                    metadata=metadata,
                    chunk_info={
                        'chunk_number': chunk_number,
                        'total_chunks': total_chunks,
                        'timestamp': datetime.now().isoformat(),
                        'words_in_chunk': len(chunk)
                    })
            
            # Log progress
            memory_usage = self.get_memory_usage()
            logger.info(f"Saved chunk {chunk_number}/{total_chunks} to {chunk_file}")
            logger.info(f"Current memory usage: {memory_usage:.2f} GB")
            
        logger.info(f"Successfully saved all {len(processed_data)} embeddings in {total_chunks} chunks")

    def compile_chunks(self, output_dir: str = "datapipeline/data/output", cleanup: bool = True) -> bool:
        """Compile all chunks into a single file and optionally cleanup"""
        output_dir = Path(output_dir)
        
        try:
            # Find all chunk files
            chunk_files = sorted(output_dir.glob("embeddings_chunk_*_of_*.npz"))
            if not chunk_files:
                logger.error("No chunk files found to compile")
                return False
            
            logger.info(f"Found {len(chunk_files)} chunks to compile")
            
            # Initialize lists to store all data
            all_embeddings = []
            all_metadata = []
            total_words = 0
            
            # Load each chunk
            for chunk_file in tqdm(chunk_files, desc="Compiling chunks"):
                chunk_data = np.load(chunk_file, allow_pickle=True)
                all_embeddings.append(chunk_data['embeddings'])
                all_metadata.extend(chunk_data['metadata'])
                total_words += len(chunk_data['embeddings'])
            
            # Combine all embeddings
            combined_embeddings = np.concatenate(all_embeddings)
            
            # Save combined file
            combined_file = output_dir / "embeddings_combined.npz"
            logger.info(f"Saving combined file to {combined_file}")
            
            np.savez_compressed(combined_file,
                    embeddings=combined_embeddings,
                    metadata=all_metadata,
                    info={
                        'total_words': total_words,
                        'timestamp': datetime.now().isoformat(),
                        'original_chunks': len(chunk_files)
                    })
            
            # Cleanup if requested
            if cleanup:
                logger.info("Cleaning up chunk files...")
                for chunk_file in chunk_files:
                    chunk_file.unlink()
                logger.info("Chunk files removed")
            
            logger.info(f"Successfully compiled {total_words} embeddings into {combined_file}")
            logger.info(f"Combined file size: {combined_file.stat().st_size / (1024*1024*1024):.2f} GB")
            return True
            
        except Exception as e:
            logger.error(f"Error compiling chunks: {str(e)}")
            return False

if __name__ == "__main__":
    print("Starting BERTy processing...")
    logger.info("Initializing BERTy...")
    berty = BERTy()
    
    input_file = "datapipeline/data/input/combined_language_data_with_translation_cleaned.csv"
    print(f"Attempting to process file: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        print(f"ERROR: File not found: {input_file}")
        exit(1)
        
    try:
        # Read and display CSV structure
        print("Reading CSV file...")
        df = pd.read_csv(input_file)
        print(f"Found {len(df)} rows in CSV")
        print(f"Columns: {df.columns.tolist()}")
        
        logger.info(f"CSV columns found: {df.columns.tolist()}")
        logger.info(f"Total rows to process: {len(df)}")
        
        # Show first few rows of data
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Process the data and get enriched DataFrame
        enriched_df = berty.process_csv_file(input_file)
        logger.info(f"Successfully processed {len(enriched_df)} words")
        
        # Print sample results
        if not enriched_df.empty:
            sample = enriched_df.iloc[0]
            logger.info("\nSample processed data:")
            logger.info(f"Word: {sample['word']}")
            logger.info(f"Language: {sample['language']}")
            logger.info(f"Translation: {sample['english_translation']}")
            logger.info(f"Time period: {sample['time_period']}")
            logger.info(f"Vector shape: {len(sample['vector'])}")
            logger.info(f"Vector preview: {sample['vector'][:5]}...")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        logger.error(f"Error processing file: {str(e)}")
        raise

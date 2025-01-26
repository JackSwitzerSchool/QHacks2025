import pandas as pd
import json
from pathlib import Path
from typing import Union, List, Dict
from abc import ABC, abstractmethod
import logging
import os
import sys
import requests
import re
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wikiscrape import WikiScraper
from datetime import datetime
import time
from parser import PIEParser, OldEnglishParser, LatinParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datapipeline/data/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Reduce noise from HTTP libraries even further
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for different data sources"""
    
    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        pass

class CSVDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_data(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        with open('datapipeline/config.json', 'r') as f:
            config = json.load(f)
        required_fields = config['DATA_SCHEMA']['required_fields']
        return all(field in df.columns for field in required_fields)

class ExcelDataSource(DataSource):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_data(self) -> pd.DataFrame:
        try:
            return pd.read_excel(self.file_path)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        with open('datapipeline/config.json', 'r') as f:
            config = json.load(f)
        required_fields = config['DATA_SCHEMA']['required_fields']
        return all(field in df.columns for field in required_fields)

class WikiDataSource(DataSource):
    """Data source for Wiktionary data"""
    
    def __init__(self, category: str):
        self.category = category
        self.scraper = WikiScraper()

    def read_data(self, limit: int = None, timeout: int = None) -> pd.DataFrame:
        try:
            pages = self.scraper.download_pie_roots(limit=limit)
            
            data = []
            for page in pages:
                extracted = self.scraper.extract_pie_data(page)
                if extracted:  # Include any page with extracted data
                    data.append(extracted)
            
            df = pd.DataFrame(data)
            
            # Show what we found
            logger.info("\nExtracted data summary:")
            logger.info(f"Number of entries: {len(df)}")
            logger.info(f"Columns found: {df.columns.tolist()}")
            if not df.empty:
                logger.info("\nSample entry:")
                logger.info(df.iloc[0].to_dict())
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading Wiktionary data: {e}")
            raise

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate against required fields from config"""
        with open('datapipeline/config.json', 'r') as f:
            config = json.load(f)
        required_fields = config['DATA_SCHEMA']['required_fields']
        
        # If we have raw data, transform it to match schema
        if 'raw_content' in df.columns:
            df['ipa_phoneme'] = None
            df['english_translation'] = None
            df['description'] = None
            # We already have original_characters
        
        has_fields = all(field in df.columns for field in required_fields)
        logger.info(f"Schema validation - Required fields: {required_fields}")
        logger.info(f"Schema validation - Present fields: {df.columns.tolist()}")
        logger.info(f"Schema validation result: {has_fields}")
        return has_fields

    def download_old_english_terms(self, limit: int = None) -> List[Dict]:
        """Download Old English terms with IPA pronunciations"""
        logger.info("Starting bulk download of Old English terms")
        pages = []
        continue_param = None
        batch_count = 0
        
        while True:
            try:
                # Implement batch delays
                if batch_count > 0:
                    logger.info(f"Waiting {self.scraper.batch_delay}s between batches...")
                    time.sleep(self.scraper.batch_delay)
                
                params = {
                    'action': 'query',
                    'format': 'json',
                    'generator': 'categorymembers',
                    'gcmtitle': "Category:Old_English_terms_with_IPA_pronunciation",
                    'gcmlimit': min(self.scraper.batch_size, limit - len(pages) if limit else self.scraper.batch_size),
                    'prop': 'revisions',
                    'rvprop': 'content|ids|timestamp',
                    'rvslots': 'main'
                }
                
                if continue_param:
                    params.update(continue_param)
                
                # Make request with retry logic
                for attempt in range(self.scraper.max_retries):
                    try:
                        response = self.scraper.session.get(self.scraper.BASE_API_URL, params=params)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt < self.scraper.max_retries - 1:
                            logger.warning(f"Request failed (attempt {attempt + 1}/{self.scraper.max_retries}), waiting {self.scraper.retry_delay}s...")
                            time.sleep(self.scraper.retry_delay)
                        else:
                            logger.error(f"Failed after {self.scraper.max_retries} attempts: {e}")
                            time.sleep(self.scraper.error_delay)
                            raise
                
                # Process response
                if 'query' in data and 'pages' in data['query']:
                    new_pages = list(data['query']['pages'].values())
                    pages.extend(new_pages)
                    batch_count += 1
                    
                    logger.info(f"Batch {batch_count}: Retrieved {len(new_pages)} pages (Total: {len(pages)})")
                    
                    # Check if we've hit the limit
                    if limit and len(pages) >= limit:
                        pages = pages[:limit]
                        break
                    
                    # Check for more pages
                    if 'continue' in data:
                        continue_param = data['continue']
                    else:
                        break
                else:
                    break
                
                time.sleep(self.scraper.delay)
                
            except Exception as e:
                logger.error(f"Error in batch download: {e}")
                break
        
        logger.info(f"Downloaded {len(pages)} Old English terms")
        return pages

    def extract_old_english_data(self, page_data: Dict) -> Dict:
        """Extract Old English term data with IPA pronunciation"""
        try:
            if 'revisions' not in page_data or not page_data['revisions']:
                return {}
            
            content = page_data['revisions'][0]['slots']['main']['*']
            title = page_data.get('title', '')
            
            # Extract Old English section
            oe_section = re.search(r'==\s*Old English\s*==\s*(.*?)(?=\n==[^=]|\Z)', content, re.DOTALL)
            if not oe_section:
                logger.debug(f"No Old English section found in {title}")
                return {}
            
            # Prepare raw data for parsing
            raw_data = {
                'title': title,
                'raw_content': oe_section.group(1),
                'original_characters': title
            }
            
            # Use Claude to parse the content
            parser = OldEnglishParser()
            parsed = parser.parse_entry(raw_data)
            
            # Combine results
            result = {
                'title': title,
                'original_characters': title,
                'ipa_phoneme': parsed.get('ipa_phoneme'),
                'english_translation': parsed.get('english_translation'),
                'raw_content': oe_section.group(1)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting Old English data from {page_data.get('title', 'unknown')}: {e}")
            return {}

    def download_latin_terms(self, limit: int = None) -> List[Dict]:
        """Download Latin terms with IPA pronunciation"""
        category = "Category:Latin_terms_with_IPA_pronunciation"
        pages = []
        
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": self.scraper.batch_size,
                "cmtype": "page"
            }
            
            if limit:
                params["cmlimit"] = min(limit, self.scraper.batch_size)
            
            while True:
                response = self.scraper.session.get(self.scraper.BASE_API_URL, params=params)
                if not response:
                    break
                    
                data = response.json()
                batch = data["query"]["categorymembers"]
                pages.extend(batch)
                
                if limit and len(pages) >= limit:
                    pages = pages[:limit]
                    break
                    
                if "continue" not in data:
                    break
                    
                params["cmcontinue"] = data["continue"]["cmcontinue"]
                time.sleep(self.scraper.batch_delay)
            
            # Get full page content for each term
            content_pages = self.scraper._get_page_contents([p["title"] for p in pages])
            return content_pages
            
        except Exception as e:
            logger.error(f"Error downloading Latin terms: {e}")
            return []

    def extract_latin_data(self, page_data: Dict) -> Dict:
        """Extract Latin term data with IPA pronunciation"""
        try:
            if 'revisions' not in page_data or not page_data['revisions']:
                return {}
            
            content = page_data['revisions'][0]['slots']['main']['*']
            title = page_data.get('title', '')
            
            # Extract Latin section
            latin_section = re.search(r'==Latin==\s*(.*?)(?=\n==|\Z)', content, re.DOTALL)
            if not latin_section:
                return {}
            
            latin_content = latin_section.group(1)
            
            # Extract IPA
            ipa_match = re.search(r'\* IPA(?:\(key\))?: /([^/]+)/', latin_content)
            pronunciation = ipa_match.group(1) if ipa_match else None
            
            # Extract part of speech and definition
            pos_section = re.search(r'===(?:Noun|Verb|Adjective|Adverb|Pronoun)===\s*#\s*(.+?)(?=\n[^#]|\Z)', latin_content, re.DOTALL)
            definition = pos_section.group(1).strip() if pos_section else None
            
            return {
                'title': title,
                'original_characters': title,
                'ipa_phoneme': pronunciation,
                'english_translation': definition,
                'raw_content': latin_content
            }
            
        except Exception as e:
            logger.error(f"Error extracting Latin data from {page_data.get('title', 'unknown')}: {e}")
            return {}

class DataPreprocessor:
    """Handles data preprocessing and standardization"""
    
    def __init__(self):
        with open('datapipeline/config.json', 'r') as f:
            self.config = json.load(f)

    def standardize_ipa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize IPA notation"""
        # TODO: Implement IPA standardization
        return df

    def validate_translations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and potentially correct translations using AI"""
        # TODO: Implement translation validation
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        # Remove leading/trailing whitespace
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Remove duplicate entries
        df = df.drop_duplicates()
        
        return df

class LanguagePipeline:
    """Main pipeline class orchestrating the data processing"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self._setup_directories()

    def _setup_directories(self):
        """Create input and output directories if they don't exist"""
        dirs = [
            self.preprocessor.config['INPUT_DIR'],
            self.preprocessor.config['OUTPUT_DIR'],
            'datapipeline/data/raw'  # Add raw data directory
        ]
        for dir_path in dirs:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {path}")

    def process_language_terms(self, source: WikiDataSource, parser_class, language: str, batch_size: int) -> pd.DataFrame:
        """Process terms for a specific language with specified batch size"""
        try:
            # Get raw data
            pages = source.scraper.download_latin_terms(limit=batch_size)
            if not pages:
                logger.warning("No pages downloaded")
                return pd.DataFrame()
            
            logger.info(f"Downloaded {len(pages)} pages")
            
            # Extract raw data
            raw_data = []
            for page in pages:
                extracted = source.scraper.extract_latin_data(page)
                if extracted:
                    raw_data.append(extracted)
            
            return self.process_batch(raw_data, parser_class, language)
            
        except Exception as e:
            logger.error(f"Error processing {language} terms: {str(e)}")
            return pd.DataFrame()

    def process_batch(self, raw_data: List[Dict], parser_class, language: str) -> pd.DataFrame:
        """Process a batch of raw data through the parser"""
        try:
            # Process with Haiku
            parser = parser_class()
            parsed_data = parser.parse_batch(raw_data, len(raw_data))
            
            if not parsed_data:
                logger.warning("No data parsed")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(parsed_data)
            
            # Save raw data for this batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_df = pd.DataFrame(raw_data)
            raw_path = Path(self.preprocessor.config['INPUT_DIR']) / f"{language}_batch_{timestamp}_raw.csv"
            raw_df.to_csv(raw_path, index=False, encoding='utf-8')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return pd.DataFrame()

    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data with timestamp"""
        # Save processed data
        output_path = Path(self.preprocessor.config['OUTPUT_DIR']) / f"{filename}_processed.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Processed data saved to {output_path}")

        # Save raw data
        raw_path = Path(self.preprocessor.config['INPUT_DIR']) / f"{filename}_raw.csv"
        df.to_csv(raw_path, index=False, encoding='utf-8')
        logger.info(f"Raw data saved to {raw_path}")

def main():
    pipeline = LanguagePipeline()
    batch_size = 100  # Process 100 entries at a time
    haiku_batch_size = 20  # 5 parallel calls of 20 each
    
    print("\nStarting Latin terms processing pipeline...")
    
    # Load from cache
    source = WikiDataSource("Latin_terms_with_IPA_pronunciation")
    cache_path = source.scraper._get_cache_path("Latin_terms_with_IPA_pronunciation")
    
    if not cache_path.exists():
        print("No cache found. Run the scraper first to build cache.")
        return
        
    with open(cache_path, 'rb') as f:
        all_pages = pickle.load(f)
    
    total_pages = len(all_pages)
    print(f"\nTotal cached Latin terms: {total_pages}")
    
    # Process in batches
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"Latin_terms_full_{timestamp}"
    all_results = []
    
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        current_pages = all_pages[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_pages + batch_size - 1)//batch_size}")
        print(f"Pages {batch_start} to {batch_end}")
        
        # Extract data for this batch
        extracted_data = []
        for page in current_pages:
            data = source.scraper.extract_latin_data(page)
            if data:
                extracted_data.append(data)
        
        # Process all 5 Haiku calls for this batch
        batch_results = []
        for i in range(0, len(extracted_data), haiku_batch_size):
            haiku_batch = extracted_data[i:i + haiku_batch_size]
            processed_data = pipeline.process_batch(
                haiku_batch,
                parser_class=LatinParser,
                language="latin"
            )
            if not processed_data.empty:
                batch_results.append(processed_data)
        
        # Add batch results to overall results
        if batch_results:
            batch_df = pd.concat(batch_results, ignore_index=True)
            all_results.append(batch_df)
            print(f"Processed {len(batch_df)} entries (Total: {sum(len(df) for df in all_results)})")
            print(f"IPA success rate: {(batch_df['ipa_phoneme'].notna().sum() / len(batch_df)) * 100:.1f}%")
            print(f"Translation success rate: {(batch_df['english_translation'].notna().sum() / len(batch_df)) * 100:.1f}%")
        
        # Save after every 5 batches
        if len(all_results) >= 5:
            combined_df = pd.concat(all_results, ignore_index=True)
            pipeline.save_processed_data(combined_df, output_file)
            print(f"\nSaved {len(combined_df)} entries to file")
            all_results = []  # Clear the results list

    # Save any remaining results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        pipeline.save_processed_data(combined_df, output_file)
        print(f"\nSaved final {len(combined_df)} entries to file")

    # Print final summary
    final_df = pd.read_csv(Path(pipeline.preprocessor.config['OUTPUT_DIR']) / f"{output_file}_processed.csv")
    print("\nFinal Processing Summary:")
    print(f"Total entries processed: {len(final_df)}")
    print(f"Unique terms: {final_df['original_characters'].nunique()}")
    print(f"IPA success rate: {(final_df['ipa_phoneme'].notna().sum() / len(final_df)) * 100:.1f}%")
    print(f"Translation success rate: {(final_df['english_translation'].notna().sum() / len(final_df)) * 100:.1f}%")

if __name__ == "__main__":
    main()

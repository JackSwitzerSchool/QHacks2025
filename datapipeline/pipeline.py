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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wikiscrape import WikiScraper
from datetime import datetime
import time
from parser import PIEParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datapipeline/data/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Reduce logging from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
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
            
            # Extract Old English section and IPA
            oe_section = re.search(r'==Old English==\s*(.*?)(?=\n==|\Z)', content, re.DOTALL)
            if not oe_section:
                return {}
            
            oe_content = oe_section.group(1)
            
            # Extract IPA
            ipa_match = re.search(r'\* IPA(?:\(key\))?: /([^/]+)/', oe_content)
            pronunciation = ipa_match.group(1) if ipa_match else None
            
            # Extract definition/meaning
            definition_section = re.search(r'===(?:Noun|Verb|Adjective|Adverb)===\s*#\s*(.+?)(?=\n[^#]|\Z)', oe_content, re.DOTALL)
            definition = definition_section.group(1).strip() if definition_section else None
            
            return {
                'title': title,
                'original_characters': title,
                'ipa_phoneme': pronunciation,
                'english_translation': definition,
                'raw_content': oe_content
            }
            
        except Exception as e:
            logger.error(f"Error extracting Old English data from {page_data.get('title', 'unknown')}: {e}")
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
        # Create data directories if they don't exist
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

    def process_source(self, source: DataSource, limit: int = None, timeout: int = None) -> pd.DataFrame:
        """Process a single data source with two-phase parsing"""
        if isinstance(source, WikiDataSource):
            # Check for cached data first
            cache_dir = Path('datapipeline/data/raw')
            cached_files = list(cache_dir.glob('pie_roots_raw_*.csv'))
            
            use_cache = False
            if cached_files:
                latest_cache = max(cached_files, key=lambda x: x.stat().st_mtime)
                try:
                    # Try to read the cache file
                    raw_df = pd.read_csv(latest_cache)
                    if len(raw_df) > 0 and 'title' in raw_df.columns:
                        print(f"\nUsing cached data from {latest_cache.name}")
                        print(f"Found {len(raw_df)} entries in cache")
                        print(f"Columns in cache: {raw_df.columns.tolist()}")
                        print("\nSample entries:")
                        for _, row in raw_df.head(3).iterrows():
                            print(f"- {row['title']}")
                        use_cache = True
                    else:
                        print(f"\nCache file {latest_cache.name} appears to be empty or corrupted")
                except Exception as e:
                    print(f"\nError reading cache file {latest_cache.name}: {e}")
            
            if not use_cache:
                # Download fresh data
                print("\nDownloading fresh data...")
                raw_pages = source.scraper.download_pie_roots(limit=limit)
                raw_data = []
                for page in raw_pages:
                    extracted = source.scraper.extract_raw_pie_data(page)
                    if extracted:
                        raw_data.append(extracted)
                
                raw_df = pd.DataFrame(raw_data)
                logger.info(f"Downloaded {len(raw_df)} raw entries")
                
                # Save raw data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cache_path = f'datapipeline/data/raw/pie_roots_raw_{timestamp}.csv'
                raw_df.to_csv(cache_path, index=False)
                print(f"Saved fresh data to {cache_path}")
            
            # Parse with Claude
            try:
                parser = PIEParser()
                parsed_data = []
                success_count = 0
                error_count = 0
                
                print(f"\nProcessing {len(raw_df)} entries...")
                for idx, row in enumerate(raw_df.iterrows(), 1):
                    try:
                        parsed = parser.parse_entry(row[1].to_dict())
                        parsed['original_characters'] = row[1]['original_characters']
                        parsed_data.append(parsed)
                        if any(parsed.values()):
                            success_count += 1
                        if idx % 10 == 0:  # Progress update every 10 entries
                            print(f"Processed {idx}/{len(raw_df)} entries...", end='\r')
                    except Exception as e:
                        error_count += 1
                        parsed_data.append({
                            'original_characters': row[1]['original_characters'],
                            'ipa_phoneme': None,
                            'english_translation': None,
                            'description': None
                        })
                
                print(f"\nCompleted processing with {success_count} successes and {error_count} errors")
                df = pd.DataFrame(parsed_data)
                
            except Exception as e:
                logger.error(f"Error in parsing phase: {e}")
                df = raw_df
        else:
            df = source.read_data()
        
        # Validate schema
        if not source.validate_schema(df):
            raise ValueError("Data does not match required schema")
        
        # Preprocess data
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.standardize_ipa(df)
        df = self.preprocessor.validate_translations(df)
        
        return df

    def save_processed_data(self, df: pd.DataFrame, language: str, output_format: str = 'csv'):
        """Save processed data with proper Unicode encoding"""
        # Save to output directory (processed data)
        output_dir = Path(self.preprocessor.config['OUTPUT_DIR'])
        if output_format == 'csv':
            output_path = output_dir / f"{language}_processed.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif output_format == 'excel':
            output_path = output_dir / f"{language}_processed.xlsx"
            df.to_excel(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        # Save raw data to input directory (for training)
        input_dir = Path(self.preprocessor.config['INPUT_DIR'])
        input_path = input_dir / f"{language}_raw.csv"
        df.to_csv(input_path, index=False, encoding='utf-8')
        logger.info(f"Raw data saved to {input_path}")

    def process_old_english(self, source: WikiDataSource, limit: int = None) -> pd.DataFrame:
        """Process Old English terms specifically"""
        try:
            # Download and extract data
            pages = source.download_old_english_terms(limit=limit)
            
            parsed_data = []
            success_count = 0
            error_count = 0
            
            for page in pages:
                try:
                    extracted = source.extract_old_english_data(page)
                    if extracted:
                        parsed_data.append(extracted)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error processing page {page.get('title', 'unknown')}: {e}")
                    error_count += 1
            
            print(f"\nCompleted processing with {success_count} successes and {error_count} errors")
            return pd.DataFrame(parsed_data)
            
        except Exception as e:
            logger.error(f"Error in Old English processing: {e}")
            return pd.DataFrame()

def main():
    pipeline = LanguagePipeline()
    
    # Configure scraper for Old English terms
    wiki_source = WikiDataSource("Old_English_terms_with_IPA_pronunciation")
    wiki_source.scraper.delay = 1.0
    wiki_source.scraper.batch_size = 50
    wiki_source.scraper.batch_delay = 3
    
    print("\nStarting Old English term extraction...")
    processed_data = pipeline.process_old_english(wiki_source, limit=None)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Old_English_terms_{timestamp}"
    pipeline.save_processed_data(processed_data, filename)
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total entries: {len(processed_data)}")
    
    for field in ['ipa_phoneme', 'english_translation']:
        present = processed_data[field].notna()
        print(f"\n{field.replace('_', ' ').title()}:")
        print(f"- Count: {present.sum()} ({present.mean():.1%})")
        if present.any():
            print("- Sample values:")
            for val in processed_data[field][present].head():
                print(f"  * {val[:100]}...")

if __name__ == "__main__":
    main()

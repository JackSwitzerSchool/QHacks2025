PIE_WIKI = "https://en.wiktionary.org/wiki/Category:Proto-Indo-European_roots"
PIE_ENGLISH = "https://en.wiktionary.org/w/index.php?title=Category:Old_English_terms_with_IPA_pronunciation&pagefrom=ACWELEST%0Aacwelest#mw-pages"
OLD_ENGLISH_CATEGORY = "Category:Old_English_terms_with_IPA_pronunciation"

import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional
import time
import re

logger = logging.getLogger(__name__)

class WikiScraper:
    """Handles interaction with Wiktionary API and web scraping"""
    
    BASE_API_URL = "https://en.wiktionary.org/w/api.php"
    BASE_URL = "https://en.wiktionary.org"
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize WikiScraper with Wiktionary-friendly settings
        
        Args:
            delay: Time to wait between requests (default 1.0 seconds)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LanguageResearchBot/1.0 (Etymology Research Project; Contact: your@email.com)',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        self.delay = delay
        self.max_retries = 3
        self.retry_delay = 5
        self.batch_size = 50  # Wiktionary recommends smaller batches
        self.batch_delay = 3  # Delay between batches
        self.error_delay = 30  # Delay after errors
        
        logger.info(f"WikiScraper initialized with:")
        logger.info(f"- Request delay: {self.delay}s")
        logger.info(f"- Batch size: {self.batch_size}")
        logger.info(f"- Batch delay: {self.batch_delay}s")

    def download_pie_roots(self, limit: int = None) -> List[Dict]:
        """Download PIE root data in bulk with rate limiting"""
        logger.info("Starting bulk download of PIE roots")
        pages = []
        continue_param = None
        batch_count = 0
        
        while True:
            try:
                # Implement batch delays
                if batch_count > 0:
                    logger.info(f"Waiting {self.batch_delay}s between batches...")
                    time.sleep(self.batch_delay)
                
                params = {
                    'action': 'query',
                    'format': 'json',
                    'generator': 'categorymembers',
                    'gcmtitle': 'Category:Proto-Indo-European_roots',
                    'gcmlimit': min(self.batch_size, limit - len(pages) if limit else self.batch_size),
                    'prop': 'revisions',
                    'rvprop': 'content|ids|timestamp',
                    'rvslots': 'main'
                }
                
                if continue_param:
                    params.update(continue_param)
                
                # Make request with retry logic
                for attempt in range(self.max_retries):
                    try:
                        response = self.session.get(self.BASE_API_URL, params=params)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), waiting {self.retry_delay}s...")
                            time.sleep(self.retry_delay)
                        else:
                            logger.error(f"Failed after {self.max_retries} attempts: {e}")
                            time.sleep(self.error_delay)
                            raise
                
                # Process response
                if 'query' in data and 'pages' in data['query']:
                    new_pages = list(data['query']['pages'].values())
                    pages.extend(new_pages)
                    batch_count += 1
                    
                    logger.info(f"Batch {batch_count}: Downloaded {len(new_pages)} pages (Total: {len(pages)})")
                    
                    if limit and len(pages) >= limit:
                        pages = pages[:limit]
                        break
                
                if 'continue' not in data:
                    break
                    
                continue_param = data['continue']
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error in download: {e}")
                time.sleep(self.error_delay)
                break
        
        logger.info(f"Completed download of {len(pages)} PIE roots in {batch_count} batches")
        return pages

    def extract_pie_data(self, page_data: Dict) -> Dict:
        try:
            if 'revisions' not in page_data or not page_data['revisions']:
                return {}
            
            content = page_data['revisions'][0]['slots']['main']['*']
            title = page_data.get('title', '')
            
            # Skip category pages and non-PIE pages
            if title.startswith('Category:') or 'Proto-Indo-European' not in title:
                return {}

            def clean_wiki_text(text: str) -> str:
                """Clean wiki markup from text"""
                if not text:
                    return ""
                # Remove references
                text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
                # Remove remaining ref tags
                text = re.sub(r'<ref[^/]*?/>', '', text)
                # Convert wiki links to plain text
                text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
                # Remove templates
                text = re.sub(r'\{\{(?:[^}])*\}\}', '', text)
                # Remove quotes and formatting
                text = re.sub(r"'''?|''?", '', text)
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            def extract_definitions() -> List[str]:
                """Extract clean definitions"""
                patterns = [
                    r'#\s*([^\n#]*?)(?=\n#|\n\n|\Z)',  # Numbered definitions
                    r'===Meaning===\s*([^=\n]+)',       # Meaning section
                    r'===Root===\s*[^#\n]*\n#\s*([^\n]+)'  # Root section definitions
                ]
                definitions = []
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.DOTALL)
                    for match in matches:
                        cleaned = clean_wiki_text(match.group(1))
                        if cleaned and len(cleaned) > 1:  # Avoid single-char matches
                            definitions.append(cleaned)
                return list(dict.fromkeys(definitions))  # Remove duplicates

            def extract_phonological_info() -> str:
                """Extract phonological information"""
                patterns = [
                    r'\{\{PIE\|([^}|]+?)(?:\||\})',
                    r'\{\{ine-root\|([^}|]+?)(?:\||\})',
                    r'\{\{etymon\|ine-pro\|([^}|]+?)(?:\||\})',
                    r'===Reconstruction===\s*([^=\n]+)'
                ]
                phonemes = []
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        cleaned = clean_wiki_text(match.group(1))
                        if cleaned and not cleaned.startswith('id=') and not cleaned.startswith('pos='):
                            phonemes.append(cleaned)
                return '; '.join(dict.fromkeys(phonemes))

            def extract_description() -> str:
                """Extract and combine relevant descriptions"""
                sections = {
                    'notes': r'====Reconstruction notes====\s*(.*?)(?=\n===|\Z)',
                    'etymology': r'===Etymology===\s*(.*?)(?=\n===|\Z)',
                    'description': r"'''Description:?'''\s*(.*?)(?=\n===|\Z)"
                }
                descriptions = []
                for section_name, pattern in sections.items():
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        cleaned = clean_wiki_text(match.group(1))
                        if cleaned:
                            descriptions.append(cleaned)
                return ' | '.join(descriptions)

            # Extract and clean data
            translations = extract_definitions()
            phoneme_info = extract_phonological_info()
            description = extract_description()

            # Build result
            extracted = {
                'original_characters': title.replace('Reconstruction:Proto-Indo-European/', '').strip(),
                'ipa_phoneme': phoneme_info if phoneme_info else None,
                'english_translation': '; '.join(translations) if translations else None,
                'description': description if description else None
            }

            # Log clean results
            for key, value in extracted.items():
                if value:
                    logger.info(f"Found clean {key}: {value[:100]}...")

            return extracted

        except Exception as e:
            logger.error(f"Error extracting data from {page_data.get('title', 'unknown')}: {e}")
            return {}

    def get_category_members(self, category: str, limit: int = None, timeout: int = None) -> List[Dict]:
        """
        Get pages in a category using the API
        
        Args:
            category: Category name (e.g., "Old_English_terms_with_IPA_pronunciation")
            limit: Maximum number of entries to fetch (None for all)
            timeout: Maximum time in seconds to spend fetching (None for no limit)
            
        Returns:
            List of dictionaries containing page information
        """
        members = []
        continue_param = None
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.info(f"Timeout reached after {timeout} seconds")
                break
            
            # Check limit
            if limit and len(members) >= limit:
                logger.info(f"Reached limit of {limit} entries")
                break
            
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': min(50, limit) if limit else 500  # Smaller batch size for limited runs
            }
            
            if continue_param:
                params.update(continue_param)
            
            try:
                response = self.session.get(self.BASE_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data and 'categorymembers' in data['query']:
                    new_members = data['query']['categorymembers']
                    if limit:
                        # Only take up to the limit
                        remaining = limit - len(members)
                        new_members = new_members[:remaining]
                    members.extend(new_members)
                
                if 'continue' not in data or (limit and len(members) >= limit):
                    break
                    
                continue_param = data['continue']
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error fetching category members: {e}")
                break
        
        return members

    def get_page_ipa(self, title: str) -> Optional[str]:
        """
        Extract IPA pronunciation from a Wiktionary page
        
        Args:
            title: Page title
            
        Returns:
            IPA pronunciation if found, None otherwise
        """
        params = {
            'action': 'parse',
            'format': 'json',
            'page': title,
            'prop': 'text'
        }
        
        try:
            response = self.session.get(self.BASE_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' in data and 'text' in data['parse']:
                html = data['parse']['text']['*']
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for IPA pronunciation in various formats
                ipa_span = soup.find('span', class_='IPA')
                if ipa_span:
                    return ipa_span.text.strip()
                    
                # Try alternative format
                ipa_element = soup.find('span', {'lang': 'en-IPA'})
                if ipa_element:
                    return ipa_element.text.strip()
                    
            time.sleep(self.delay)
            
        except Exception as e:
            logger.error(f"Error fetching IPA for {title}: {e}")
        
        return None

    def get_etymology(self, title: str) -> Optional[str]:
        """
        Extract etymology information from a Wiktionary page
        
        Args:
            title: Page title
            
        Returns:
            Etymology text if found, None otherwise
        """
        params = {
            'action': 'parse',
            'format': 'json',
            'page': title,
            'prop': 'text'
        }
        
        try:
            response = self.session.get(self.BASE_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' in data and 'text' in data['parse']:
                html = data['parse']['text']['*']
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find etymology section
                etym_heading = soup.find(['h3', 'h4'], string='Etymology')
                if etym_heading:
                    etym_content = []
                    for sibling in etym_heading.find_next_siblings():
                        if sibling.name and sibling.name.startswith('h'):
                            break
                        etym_content.append(sibling.get_text().strip())
                    return ' '.join(etym_content)
                    
            time.sleep(self.delay)
            
        except Exception as e:
            logger.error(f"Error fetching etymology for {title}: {e}")
        
        return None

    def extract_raw_pie_data(self, page_data: Dict) -> Dict:
        """Extract raw data with minimal cleaning for later processing"""
        try:
            if 'revisions' not in page_data or not page_data['revisions']:
                return {}
            
            content = page_data['revisions'][0]['slots']['main']['*']
            title = page_data.get('title', '')
            
            # Skip category pages and non-PIE pages
            if title.startswith('Category:') or 'Proto-Indo-European' not in title:
                return {}

            # Extract sections with minimal processing
            sections = {
                'root': re.search(r'===Root===\s*(.*?)(?=\n===|\Z)', content, re.DOTALL),
                'etymology': re.search(r'===Etymology===\s*(.*?)(?=\n===|\Z)', content, re.DOTALL),
                'reconstruction': re.search(r'===Reconstruction===\s*(.*?)(?=\n===|\Z)', content, re.DOTALL),
                'notes': re.search(r'====Reconstruction notes====\s*(.*?)(?=\n===|\Z)', content, re.DOTALL),
                'derived': re.search(r'====Derived terms====\s*(.*?)(?=\n===|\Z)', content, re.DOTALL)
            }

            # Build raw data object
            raw_data = {
                'title': title,
                'original_characters': title.replace('Reconstruction:Proto-Indo-European/', '').strip(),
                'raw_content': content
            }

            # Add section content if found
            for section, match in sections.items():
                if match:
                    raw_data[f'{section}_section'] = match.group(1).strip()

            return raw_data

        except Exception as e:
            logger.error(f"Error extracting raw data from {page_data.get('title', 'unknown')}: {e}")
            return {}

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
                    logger.info(f"Waiting {self.batch_delay}s between batches...")
                    time.sleep(self.batch_delay)
                
                params = {
                    'action': 'query',
                    'format': 'json',
                    'generator': 'categorymembers',
                    'gcmtitle': OLD_ENGLISH_CATEGORY,
                    'gcmlimit': min(self.batch_size, limit - len(pages) if limit else self.batch_size),
                    'prop': 'revisions',
                    'rvprop': 'content|ids|timestamp',
                    'rvslots': 'main'
                }
                
                if continue_param:
                    params.update(continue_param)
                
                # Make request with retry logic
                for attempt in range(self.max_retries):
                    try:
                        response = self.session.get(self.BASE_API_URL, params=params)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), waiting {self.retry_delay}s...")
                            time.sleep(self.retry_delay)
                        else:
                            logger.error(f"Failed after {self.max_retries} attempts: {e}")
                            time.sleep(self.error_delay)
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
                
                time.sleep(self.delay)
                
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


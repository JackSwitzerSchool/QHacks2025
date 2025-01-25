from anthropic import Anthropic
import os
from typing import Dict, List
import json
import logging
import re

logger = logging.getLogger(__name__)

class LanguageParser:
    """Base class for language-specific parsers"""
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _empty_result(self):
        """Return empty result with correct structure"""
        return {
            'ipa_phoneme': None,
            'english_translation': None,
            'original_characters': None
        }

    def _clean_ipa(self, ipa: str) -> str:
        """Clean IPA text by removing extra markers and normalizing"""
        if not ipa:
            return None
        # Remove common IPA delimiters and normalize spaces
        cleaned = re.sub(r'[/\[\]]', '', ipa).strip()
        return cleaned if cleaned else None

class LatinParser(LanguageParser):
    """Parser specifically for Latin terms"""
    
    def parse_batch(self, raw_data_list: List[Dict], batch_size: int) -> List[Dict]:
        """Parse multiple Latin entries in a single Claude request"""
        try:
            batch_prompt = "Parse multiple Latin dictionary entries. For each entry, provide:\n\n"
            batch_prompt += "ipa_phoneme: [IPA pronunciation]\n"
            batch_prompt += "english_translation: [English meaning]\n"
            batch_prompt += "part_of_speech: [grammatical category]\n\n"
            
            for i, entry in enumerate(raw_data_list, 1):
                batch_prompt += f"Entry {i}:\n"
                batch_prompt += f"Title: {entry['title']}\n"
                batch_prompt += f"Content:\n{entry.get('raw_content', '')}\n"
                batch_prompt += "\n---\n"
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                temperature=0,
                system="You are a Latin linguistics expert. Parse each entry and return the requested fields in a consistent format. Use '---' to separate entries.",
                messages=[{"role": "user", "content": batch_prompt}]
            )
            
            content = response.content[0].text
            entry_responses = content.split('---')
            parsed_entries = []
            
            for raw_data, response_text in zip(raw_data_list, entry_responses):
                parsed = self._parse_single_response(response_text.strip(), raw_data)
                parsed_entries.append(parsed)
            
            return parsed_entries
            
        except Exception as e:
            logger.error(f"Error in batch parsing: {str(e)}")
            return [self._empty_result() for _ in raw_data_list]

    def _parse_single_response(self, response_text: str, raw_data: Dict) -> Dict:
        """Parse a single Latin entry response"""
        extracted = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            for field in ['ipa_phoneme', 'english_translation', 'part_of_speech']:
                if line.lower().startswith(f"{field}:"):
                    value = line.split(':', 1)[1].strip()
                    value = value.strip('[]')
                    if value and value not in ['null', 'None', '...']:
                        extracted[field] = value
                    break

        result = {
            'ipa_phoneme': self._clean_ipa(extracted.get('ipa_phoneme', '').strip()) if 'ipa_phoneme' in extracted else None,
            'english_translation': extracted.get('english_translation', '').strip() if 'english_translation' in extracted else None,
            'part_of_speech': extracted.get('part_of_speech', '').strip() if 'part_of_speech' in extracted else None,
            'original_characters': raw_data.get('title', '').strip()
        }
        
        return result if any(result.values()) else self._empty_result()

class PIEParser(LanguageParser):
    def __init__(self):
        super().__init__()
        
    def parse_batch(self, raw_data_list: List[Dict], batch_size: int) -> List[Dict]:
        """Parse multiple entries in a single Claude request"""
        try:
            # Build combined prompt for all entries in batch
            batch_prompt = "Parse multiple Proto-Indo-European entries. For each entry, provide:\n\n"
            batch_prompt += "ipa_phoneme: [phonological reconstruction]\n"
            batch_prompt += "english_translation: [basic meaning(s)]\n"
            batch_prompt += "description: [etymological notes]\n\n"
            
            for i, entry in enumerate(raw_data_list, 1):
                batch_prompt += f"Entry {i}:\n"
                batch_prompt += f"Title: {entry['title']}\n"
                batch_prompt += self._format_entry_content(entry)
                batch_prompt += "\n---\n"
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                temperature=0,
                system="You are a PIE linguistics expert. Parse each entry and return the requested fields in a consistent format. Use '---' to separate entries.",
                messages=[{"role": "user", "content": batch_prompt}]
            )
            
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            
            # Split response into individual entries
            entry_responses = content.split('---')
            parsed_entries = []
            
            for raw_data, response_text in zip(raw_data_list, entry_responses):
                parsed = self._parse_single_response(response_text.strip(), raw_data)
                parsed_entries.append(parsed)
            
            return parsed_entries
            
        except Exception as e:
            logger.error(f"Error in batch parsing: {str(e)}")
            return [self._empty_result() for _ in raw_data_list]

    def _format_entry_content(self, entry: Dict) -> str:
        """Format entry content for batch prompt"""
        sections_text = ""
        for section in ['root', 'etymology', 'reconstruction', 'notes']:
            if f'{section}_section' in entry and entry[f'{section}_section']:
                sections_text += f"\n{section.title()} Section:\n{entry[f'{section}_section']}\n"
        return sections_text if sections_text.strip() else entry.get('raw_content', '')

    def _parse_single_response(self, response_text: str, raw_data: Dict) -> Dict:
        """Parse a single entry response"""
        extracted = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            for field in ['ipa_phoneme', 'english_translation', 'description']:
                if line.lower().startswith(f"{field}:"):
                    value = line.split(':', 1)[1].strip()
                    value = value.strip('[]')
                    if value and value not in ['null', 'None', '...']:
                        extracted[field] = value
                    break

        result = {
            'ipa_phoneme': self._clean_ipa(extracted.get('ipa_phoneme', '').strip()) if 'ipa_phoneme' in extracted else None,
            'english_translation': extracted.get('english_translation', '').strip() if 'english_translation' in extracted else None,
            'description': extracted.get('description', '').strip() if 'description' in extracted else None,
            'original_characters': raw_data.get('original_characters', raw_data.get('title', '').replace('Reconstruction:Proto-Indo-European/', '').strip())
        }
        
        return result if any(result.values()) else self._empty_result()

    def parse_entry(self, raw_data: Dict) -> Dict:
        """Use Claude to extract structured data from raw PIE entry"""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="You are a PIE linguistics expert. Return ONLY the requested fields in the exact format shown.",
                messages=[{"role": "user", "content": self._build_extraction_prompt(raw_data)}]
            )
            
            # Get the actual text content from the response
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            print(f"\nFirst pass response for {raw_data['title']}:\n{content}\n")
            
            # Parse the structured response
            extracted = {}
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                for field in ['ipa_phoneme', 'english_translation', 'description']:
                    if line.lower().startswith(f"{field}:"):
                        value = line.split(':', 1)[1].strip()
                        # Remove square brackets if present
                        value = value.strip('[]')
                        if value and value not in ['null', 'None', '...']:
                            extracted[field] = value
                        break

            if extracted:
                result = {
                    'ipa_phoneme': self._clean_ipa(extracted.get('ipa_phoneme', '').strip()) if 'ipa_phoneme' in extracted else None,
                    'english_translation': extracted.get('english_translation', '').strip() if 'english_translation' in extracted else None,
                    'description': extracted.get('description', '').strip() if 'description' in extracted else None
                }
                
                if any(result.values()):
                    print(f"Successfully parsed {raw_data['title']}: {result}")
                    return result
            
            print(f"Failed to extract data for {raw_data['title']}")
            return self._empty_result()
            
        except Exception as e:
            logger.error(f"Error parsing {raw_data['title']}: {str(e)}")
            return self._empty_result()

    def _build_extraction_prompt(self, raw_data: Dict) -> str:
        """Build the initial extraction prompt"""
        sections_text = ""
        for section in ['root', 'etymology', 'reconstruction', 'notes']:
            if f'{section}_section' in raw_data and raw_data[f'{section}_section']:
                sections_text += f"\n{section.title()} Section:\n{raw_data[f'{section}_section']}\n"
        
        return f"""Extract information from this Proto-Indo-European root entry.
Return ONLY these three fields with their values:

ipa_phoneme: [phonological reconstruction]
english_translation: [basic meaning(s)]
description: [etymological notes]

Entry to analyze:
Root: {raw_data['title']}

Content:
{sections_text if sections_text.strip() else raw_data.get('raw_content', '')}

Important:
1. Return ONLY the three fields above
2. Use exactly those field names
3. Include the colon after each field name
4. Put each field on its own line
5. Leave empty fields blank"""

class OldEnglishParser:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def parse_entry(self, raw_data: Dict) -> Dict:
        """Use Claude to extract structured data from raw Old English entry"""
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="You are an Old English linguistics expert. Extract the IPA pronunciation and meaning from Wiktionary entries. Return ONLY the requested fields in the exact format shown.",
                messages=[{"role": "user", "content": self._build_extraction_prompt(raw_data)}]
            )
            
            # Get the actual text content from the response
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            print(f"\nParsed response for {raw_data['title']}:\n{content}\n")
            
            # Parse the structured response
            extracted = {}
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            for line in lines:
                for field in ['ipa_phoneme', 'english_translation']:
                    if line.lower().startswith(f"{field}:"):
                        value = line.split(':', 1)[1].strip()
                        # Remove square brackets if present
                        value = value.strip('[]')
                        if value and value not in ['null', 'None', '...']:
                            extracted[field] = value
                        break

            if extracted:
                return {
                    'ipa_phoneme': extracted.get('ipa_phoneme', '').strip() if 'ipa_phoneme' in extracted else None,
                    'english_translation': extracted.get('english_translation', '').strip() if 'english_translation' in extracted else None
                }
            
            print(f"Failed to extract data for {raw_data['title']}")
            return self._empty_result()
            
        except Exception as e:
            logger.error(f"Error parsing {raw_data['title']}: {str(e)}")
            return self._empty_result()

    def _empty_result(self):
        """Return empty result with correct structure"""
        return {
            'ipa_phoneme': None,
            'english_translation': None
        }

    def _build_extraction_prompt(self, raw_data: Dict) -> str:
        """Build the extraction prompt"""
        return f"""Extract the IPA pronunciation and English translation from this Old English Wiktionary entry.
Return ONLY these fields with their values:

ipa_phoneme: [IPA pronunciation]
english_translation: [English meaning/translation]

Entry to analyze:
Title: {raw_data['title']}

Content:
{raw_data.get('raw_content', '')}

Important:
1. Return ONLY the two fields above
2. Use exactly those field names
3. Include the colon after each field name
4. Put each field on its own line
5. Leave empty fields blank
6. For IPA, include only the phonetic transcription without slashes or brackets
7. For translation, provide a clear, concise English meaning""" 
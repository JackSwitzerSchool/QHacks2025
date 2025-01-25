from anthropic import Anthropic
import os
from typing import Dict
import json
import logging
import re

logger = logging.getLogger(__name__)

class PIEParser:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
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

    def _empty_result(self):
        """Return empty result with correct structure"""
        return {
            'ipa_phoneme': None,
            'english_translation': None,
            'description': None
        }

    def _clean_ipa(self, text: str) -> str:
        """Clean and validate IPA text"""
        if not text:
            return None
        
        # Comprehensive PIE phoneme characters
        valid_chars = set(
            # Basic vowels and diacritics
            'aāăâeēĕêiīĭîoōŏôuūŭûəɨʉɯɪʊɛœɔæɐɑɒʌ' +
            # Basic consonants
            'bʰdʰgʰkʰpʰtʰ' +  # Aspirated stops
            'bdgptkʔmnŋlrjwʍʋfvszxɦ' +  # Other consonants
            # Special PIE characters
            'ǵḱĝǵʰḱʰĝʰ' +  # Palatals and aspirated palatals
            'gʷkʷgʷʰkʷʰ' +  # Labiovelars and aspirated labiovelars
            'ʰʷʲˠˤ' +  # Superscript modifiers
            '₁₂₃₄ᵃᵉᵒⁱ' +  # Subscript numbers and vowels
            'H' +  # Laryngeal placeholder
            '*-' +  # Reconstruction marker and hyphen
            'm̥n̥l̥r̥'  # Syllabic resonants
        )
        
        # Remove any characters not in our valid set
        cleaned = ''.join(c for c in text if c in valid_chars)
        
        # Validate basic PIE phoneme patterns
        if not cleaned or cleaned.isspace():
            return None
        
        return cleaned

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
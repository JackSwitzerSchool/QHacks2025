import os
import logging
from typing import Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class ClaudeService:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        # Load Toronto slang reference
        with open('datapipeline/data/TorontoSlangList.md', 'r') as f:
            self.toronto_slang_reference = f.read()
        
    def translate_to_toronto_slang(self, text: str) -> Optional[str]:
        """
        Two-stage translation:
        1. Convert to authentic Toronto slang
        2. Add phonetic markers for proper Toronto accent
        """
        try:
            # Stage 1: Translate to Toronto slang
            slang_prompt = f"""You are a Toronto slang and accent expert. Use this reference of Toronto slang terms and expressions:

{self.toronto_slang_reference}

Task: Translate this text into authentic Toronto slang, considering these rules:
1. Use appropriate slang terms from the reference
2. Maintain the core meaning while making it sound natural
3. Consider common Toronto phonetic patterns:
   - "th" often becomes "d" (e.g., "that" → "dat")
   - Final "g" in "-ing" words often dropped
   - "about" becomes "aboot"
   - Emphasis on first syllable
   - Rising intonation at sentence end

Text to translate: "{text}"

Provide only the translated text with no explanation:"""

            # Get slang translation
            slang_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": slang_prompt
                }]
            )
            
            toronto_text = slang_response.content[0].text.strip()
            
            # Stage 2: Add phonetic markers
            phonetic_prompt = f"""You are a Toronto accent expert. Convert this Toronto slang text into IPA notation, 
            maintaining authentic Toronto pronunciation features:

1. Key Toronto accent features to include:
   - Canadian raising: "about" → /əˈbʌʊt/
   - T-flapping between vowels
   - The "Canadian Shift" in vowels
   - Rising intonation (mark with ↗ at end of sentences)
   - Stress patterns typical of Toronto speech
   - "th" → "d" in casual speech
   - Dropped "g" in "-ing" words → /ɪn/

Toronto slang text: "{toronto_text}"

Provide only the IPA transcription with Toronto accent features. Include rising intonation marker ↗ where appropriate. Do NOT include any explanations or notes."""

            # Get phonetic translation
            phonetic_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.5,  # Lower temperature for more consistent phonetics
                messages=[{
                    "role": "user",
                    "content": phonetic_prompt
                }]
            )
            
            # Extract the IPA notation from the response
            ipa_text = phonetic_response.content[0].text.strip()
            
            # Return both translations for frontend use
            return {
                "toronto_text": toronto_text,
                "ipa_text": ipa_text
            }
            
        except Exception as e:
            logger.error(f"Claude translation failed: {str(e)}")
            return None 
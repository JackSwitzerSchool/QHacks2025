import boto3
import os
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PollyService:
    def __init__(self):
        self.polly = boto3.client('polly', region_name='us-east-1')
    
    def synthesize_ipa(self, ipa_text: str) -> Optional[str]:
        """
        Convert IPA text to speech using AWS Polly
        Returns base64 encoded audio data
        """
        try:
            # Format IPA text as SSML
            ssml_text = f'''
            <speak>
                <phoneme alphabet="ipa" ph="{ipa_text}"></phoneme>
            </speak>
            '''
            
            # Generate speech
            response = self.polly.synthesize_speech(
                Text=ssml_text,
                TextType='ssml',
                VoiceId='Joanna',
                OutputFormat='mp3',
                SampleRate='22050'
            )
            
            # Convert audio stream to base64 for frontend
            if "AudioStream" in response:
                audio_data = response['AudioStream'].read()
                return base64.b64encode(audio_data).decode('utf-8')
                
            return None
            
        except Exception as e:
            logger.error(f"Polly synthesis failed: {str(e)}")
            return None 
import boto3

polly = boto3.client('polly', region_name='us-east-1')

# Phonetic text (using IPA format or custom phoneme representation)
phonetic_text = '''
<speak>
    <phoneme alphabet="ipa" ph="/ˈ˦sɛ/"> </phoneme>.
</speak>
'''

response = polly.synthesize_speech(
    Text=phonetic_text,
    TextType='ssml',  # SSML for phonetic interpretation
    VoiceId='Joanna',  # Choose voice (adjust as necessary)
    OutputFormat='mp3',  # Desired audio format
    SampleRate='22050'
)


# Save PCM data
with open('output.mp3', 'wb') as file:
    file.write(response['AudioStream'].read())
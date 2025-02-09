import logging
import os
import base64
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """	
    Simplified function to record audio from the microphone and save it as an MP3 file.
    
    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for audio input in seconds.
    phrase_time_limit (int): Maximum time for a phrase to be spoken. If None, it will be set to the timeout value.
    """	
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    recognizer = sr.Recognizer()  # Fixed typo in variable name
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting the Background Noise.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logging.info("Please speak into the microphone...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # First save as WAV
            wav_path = file_path.replace('.mp3', '.wav')
            with open(wav_path, 'wb') as f:
                f.write(audio.get_wav_data())
            
            # Then convert to MP3
            audio_segment = AudioSegment.from_wav(wav_path)
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            # Clean up the temporary WAV file
            os.remove(wav_path)
            
            logging.info(f"Audio saved to {file_path}")
            
    except Exception as e:
        logging.error(f"Error recording audio: {e}")
        raise  # Re-raise the exception to see the full error trace

# Test the function
if __name__ == "__main__":
    audio_file_path = r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\patient_voices\patient_audio.mp3"
    record_audio(file_path=audio_file_path)
    
# Setup Speech to Text-STT-model for Transcription
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def transcribe_with_groq(stt_model, audio_file_path):
    client = Groq(api_key = GROQ_API_KEY)
    stt_model = "whisper-large-v3"
    audio_file = open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
        model=stt_model,
        file = audio_file,
        language="en",
    )

    # Print the transcription   
    return transcription.text

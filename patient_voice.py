
import logging
import os
from groq import Groq
import speech_recognition as sr
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def record_audio(file_path=r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\patient_voices\patient_audio.mp3", timeout=20, phrase_time_limit=None):
    """
    Record audio from the microphone and save it as an MP3 file.

    Args:
        file_path (str): Path to save the recorded audio.
        timeout (int): Max time to wait for audio input in seconds.
        phrase_time_limit (int): Max time for a phrase to be spoken; if None, defaults to timeout.
    """
    if phrase_time_limit is None:
        phrase_time_limit = timeout

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Recording audio. Please speak into the microphone...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Save temporary WAV file
            wav_path = file_path.replace('.mp3', '.wav')
            with open(wav_path, 'wb') as f:
                f.write(audio.get_wav_data())
            
            # Convert WAV to MP3
            audio_segment = AudioSegment.from_wav(wav_path)
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            os.remove(wav_path)
            logging.info(f"Audio saved to {file_path}")
            return file_path
            
    except Exception as e:
        logging.error(f"Error recording audio: {e}")
        raise

def transcribe_with_groq(stt_model, audio_file_path, GROQ_API_KEY):
    """
    Transcribe audio using the Groq STT model.

    Args:
        stt_model (str): The STT model name to use.
        audio_file_path (str): Path to the audio file.
        GROQ_API_KEY (str): API key for Groq.

    Returns:
        str: The transcription text.
    """
    client = Groq(api_key=GROQ_API_KEY)
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en",
            )
        return transcription.text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

# Test the function when running this module directly.
if __name__ == "__main__":
    record_audio()
    # Test transcription with a sample audio file
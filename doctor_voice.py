from opcode import opname
import os
from pydub import AudioSegment
from gtts import gTTS
from setup_multimodel_file import analyze_image_with_query
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
import platform


# ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

# def text_to_speech_with_elevenlabs(input_text ,output_filepath):
#     client = ElevenLabs(api_key = ELEVENLABS_API_KEY)
#     audio = client.generate(
#         text = input_text,
#         voice = "Callum",
#         output_format= "mp3_22050_32" ,
#         model= "eleven_turbo_v2",
#     )
#     elevenlabs.save(audio, output_filepath)
    
#     # Convert MP3 to WAV with the same name
#     wav_filepath = output_filepath.replace('.mp3', '.wav')
#     audio = AudioSegment.from_mp3(output_filepath)
#     audio.export(wav_filepath, format='wav')
   
#     os_name = platform.system()
#     try:
#         if os_name == "Darwin":
#             subprocess.run(["afplay", output_filepath])
#         elif os_name == "Linux":
#                 subprocess.run(["aplay", output_filepath])
#         elif os_name == "Windows":
#                 # Use the WAV file for Windows playback
#                 subprocess.run(["powershell", '-c', f'(New-object Media.SoundPlayer "{wav_filepath}").PlaySync();' ])
#         else:
#                 print("Unsupported OS")
#     except Exception as e:
#             print(f"Error playing audio: {e}")
    
# input_text_11 = "I am testing the voice of the doctor, which is generated by the AI model. This is a test message of ElevelLAbs."

# text_to_speech_with_elevenlabs(input_text=input_text_11,
#        output_filepath=r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\doctor_voices\output_11labs_automatic.mp3")


def text_to_speech_with_gtts(input_text, output_dir):
    # Validate input
    if not input_text or not isinstance(input_text, str):
        print("No valid text provided for text-to-speech conversion.")
        return None
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find the next available voice number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("voice") and f.endswith(".mp3")]
    if existing_files:
        numbers = [int(f.replace("voice", "").replace(".mp3", "")) for f in existing_files]
        next_number = max(numbers) + 1
    else:
        next_number = 1
    
    # Define file paths
    output_filename = f"voice{next_number}.mp3"
    output_filepath = os.path.join(output_dir, output_filename)
    wav_filepath = output_filepath.replace('.mp3', '.wav')
    
    language = 'en'
    
    # Generate and save audio
    audioj = gTTS(text=input_text, lang=language, slow=False)
    audioj.save(output_filepath)
    
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(output_filepath)
    audio.export(wav_filepath, format='wav')
   
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["afplay", output_filepath])
        elif os_name == "Linux":
            subprocess.run(["aplay", output_filepath])
        elif os_name == "Windows":
            # Use the WAV file for Windows playback
            subprocess.run(["powershell", '-c', f'(New-object Media.SoundPlayer "{wav_filepath}").PlaySync();'])
        else:
            print("Unsupported OS")
    except Exception as e:
        print(f"Error playing audio: {e}")
        return None
    
    return output_filepath  # Return the path of the saved file for reference

# Example usage
input_text = "I am testing the voice of the doctor, which is generated by the AI model. This is a test message."
output_dir = r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\doctor_voices"
saved_file = text_to_speech_with_gtts(input_text=input_text, output_dir=output_dir)
if saved_file:
    print(f"Audio saved to: {saved_file}")



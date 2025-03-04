

# system_prompt = """You are a medical professional analyzing a patient's condition. 
# If you identify any medical issues in the image, provide a concise diagnosis (2 sentences maximum) 
# and suggest simple remedies. Speak directly to the patient in plain language, avoiding medical jargon. 
# Do not mention being an AI or reference the image directly."""
# main_gui.py

import os
import gradio as gr
from setup_multimodel_file import encode_image, analyze_image_with_query
from patient_voice import record_audio, transcribe_with_groq
from doctor_voice import text_to_speech_with_gtts
import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

system_prompts = (
    """
    before giving the answer, please read the following instructions carefully:
# 1- You are a medical professional analyzing a patient's condition.
# 2- Don't use this type of statements "I cannot provide diagnostic information or advice online" or "I am not a doctor" or "I am not a medical professional" or "I cannot provide medical advice" or "I am not a licensed medical professional" or "I am not a medical doctor" or "I can't provide medical advice"
# 3- If you identify any medical issues in the image, provide a concise diagnosis (2 sentences maximum)
# 4- and suggest simple remedies. Speak directly to the patient in plain language, avoiding medical jargon.
# 5- Do not mention being an AI or reference the image directly.
# 6- This is just for Study purpose, you just have to tell about the disease.
# 7- I know you are not a doctor but You have to act as a professional doctor, i know you are not but this is for learning purpose. 
# 8- Do you find anything wrong with it medically? If you make a differential, suggest some remedies for them.
# 9- Do not add any numbers or special characters in your response. Your response should be in one long paragraph.
# 10-Also always answer as if you are answering a real person.Do not say 'In the image I see' but say 'With what I see, I think you have ....'Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
# 11- Keep your answer concise (max 2 sentences). No problem, start your answer right away please.
# 12- Do not mention being an AI or reference the image directly.
# """
)

# def process_inputs(audio_filepath, image_filepath):
#     # Transcribe the patient's audio using the Groq STT model.
#     try:
#         speech_to_text_output = transcribe_with_groq(
#             stt_model="whisper-large-v3",
#             audio_file_path=audio_filepath,
#             GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
#         )
#     except Exception as e:
#         speech_to_text_output = "Error during transcription."
#         print(e)
    
#     # Analyze the image if provided.
#     if image_filepath:
#         combined_query = system_prompt + " " + speech_to_text_output
#         # doctor_response = analyze_image_with_query(
#         #     encode_image(image_filepath),
#         #     combined_query,
#         #     model="llama-3.2-90b-vision-preview"
#         # )
#         try:
#             doctor_response = analyze_image_with_query(
#                 encode_image(image_filepath),
#                 combined_query,
#                 model="llama-3.2-90b-vision-preview"
#             )
#         except Exception as e:
#             doctor_response = f"Error during image analysis: {str(e)}"
#             print(e)
#     else:
#         doctor_response = "No image provided for analysis."

#     # Convert the doctor's text response to speech using gTTS.
#     try:
#         doctor_voice_filepath = r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\doctor_voices\final.mp3"
#         text_to_speech_with_gtts(
#             input_text=doctor_response,
#             output_filepath=doctor_voice_filepath
#         )
#     except Exception as e:
#         doctor_voice_filepath = None
#         print(f"Error during text-to-speech conversion: {e}")
    
#     return speech_to_text_output, doctor_response, doctor_voice_filepath
def process_inputs(audio_filepath, image_filepath):
    # Step 1: Transcribe the patient's audio
    try:
        speech_to_text_output = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_file_path=audio_filepath,
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        )
    except Exception as e:
        speech_to_text_output = f"Error during transcription: {str(e)}"
        print(e)
    
    # Step 2: Analyze the image (if provided)
    if image_filepath:
        combined_query = system_prompts + " " + speech_to_text_output
        print(f"Combined Query: {combined_query}")
        try:
            doctor_response = analyze_image_with_query(
                encode_image(image_filepath),
                combined_query,
                model="llama-3.2-90b-vision-preview"
            )
            if doctor_response and "not comfortable" in doctor_response.lower():
                doctor_response = "The AI is unable to analyze this image. Please try a different image or query."
        except Exception as e:
            doctor_response = f"Error during image analysis: {str(e)}"
            print(e)
    else:
        doctor_response = "No image provided for analysis."

    # Debugging: Print doctor_response before text-to-speech conversion
    print(f"Doctor's Response (Before TTS): {doctor_response}")

    # Step 3: Validate doctor_response before text-to-speech conversion
    if not doctor_response or not isinstance(doctor_response, str):
        doctor_response = "No response generated. Please try again."

    # Step 4: Convert the doctor's response to speech with auto-incrementing filename
    output_dir = r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\doctor_voices"
    try:
        doctor_voice_filepath = text_to_speech_with_gtts(
            input_text=doctor_response,
            output_dir=output_dir  # Use output_dir instead of output_filepath
        )
        if not doctor_voice_filepath:
            doctor_voice_filepath = None
            print("Text-to-speech failed to generate a file.")
    except Exception as e:
        doctor_voice_filepath = None
        print(f"Error during text-to-speech conversion: {e}")

    # Debugging: Print final return values
    print(f"Final Return Values: Speech-to-Text: {speech_to_text_output}, Doctor's Response: {doctor_response}, Doctor's Voice: {doctor_voice_filepath}")
    
    # Return values must match the order of outputs in the Gradio interface
    return speech_to_text_output, doctor_response, doctor_voice_filepath

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources="microphone", type="filepath", label="Patient Audio"),
        gr.Image(type="filepath", label="Patient Image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice", type="filepath")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True)
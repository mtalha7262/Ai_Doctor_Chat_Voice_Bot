import os
import base64
from groq import Groq  # Ensure Groq is correctly installed and imported

# Setup API Key (Replace with your actual API key or ensure it's in your environment variables)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("API key is not set. Please set GROQ_API_KEY in your environment.")

# Initialize the Groq client with the API key

# Convert image to required format (encode the image in base64)
# image_path = r"C:\Users\Muhammad_Talha\Documents\AI_Doctor_Voice_Chat_Bot\acne.jpg"  # Path to your input image
def encode_image(image_path):
    image_file = open(image_path, "rb")  
    return base64.b64encode(image_file.read()).decode('utf-8')

# Define the query for the multimodal model
query = "Is there something wrong with my face?"  # User query to the model
model = "llama-3.2-90b-vision-preview"  # Specify the model name

def analyze_image_with_query(encoded_image, query, model):  # Updated to return response
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ],
        }
    ]
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content  # Return instead of print
    except Exception as e:
        print(f"Error: {e}")
        return str(e)
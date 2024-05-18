import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from utils import record_audio, play_audio
from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64
import logging

# Load the OpenAI API key from environment variable
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
assert API_KEY, "ERROR: OpenAI Key is missing"

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global dictionary to store conversation context
global_context = {
    "conversation": False,
    "conversation_context": []
}

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process the conversation
async def start_conversation():
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        while global_context['conversation']:
            # Record audio asynchronously
            logging.info("Recording audio...")
            await loop.run_in_executor(executor, record_audio, 'test.wav')
            logging.info("Audio recording complete.")
            
            audio_file = open('test.wav', "rb")
            logging.info("Audio file opened.")

            # Process transcription and chatbot response in parallel
            transcription_future = loop.run_in_executor(executor, lambda: client.audio.transcriptions.create(model="whisper-1", file=audio_file))
            transcription = await transcription_future
            user_input = transcription.text
            logging.info(f"Transcription: {user_input}")
            st.write(f"User: {user_input}")

            # Append user input to the conversation context
            global_context['conversation_context'].append({"role": "user", "content": user_input})

            response_future = loop.run_in_executor(executor, lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=global_context['conversation_context']
            ))
            response = await response_future

            bot_response = response.choices[0].message.content
            logging.info(f"Bot response: {bot_response}")
            st.write(f"Bot: {bot_response}")

            # Append bot response to the conversation context
            global_context['conversation_context'].append({"role": "assistant", "content": bot_response})

            # Generate audio response asynchronously
            response_audio_future = loop.run_in_executor(executor, lambda: client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=bot_response
            ))
            response_audio = await response_audio_future

            # Play audio asynchronously
            await loop.run_in_executor(executor, response_audio.stream_to_file, 'output.mp3')
            await loop.run_in_executor(executor, play_audio, 'output.mp3')

# Streamlit app layout
st.set_page_config(page_title="AI Multi-modal Tutor Bot", layout="centered")

# Custom CSS for a clean black and white design
st.markdown("""
    <style>
        /* General Styling */
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        .stApp {
            background-color: #f5f5f5;
            color: #333;
        }
        /* Title Styling */
        .stMarkdown h1 {
            color: #333;
            font-weight: bold;
        }
        /* Button Styling */
        .stButton button {
            background-color: #999; /* Light grey color */
            color: #fff; /* Change text color to white */
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }
        .stButton button:active,
        .stButton button:focus {
            background-color: #bbb; /* Slightly darker light grey */
            color: #fff; /* Ensure text color stays white when clicked or focused */
        }
        .stButton button:hover {
            background-color: #bbb; /* Slightly darker light grey */
            color: #fff; /* Ensure text color stays white on hover */
        }
        /* File Uploader Styling */
        .stFileUploader label {
            font-weight: bold;
            color: #333;
        }
        /* Text Styling */
        .stMarkdown p {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI Multi-modal Tutor Bot")

# Ensure session state keys are initialized
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = False

uploaded_file = st.file_uploader("Upload an image (e.g., a math problem) to analyze before starting the conversation", type=["jpg", "jpeg", "png"])

def manage_conversation():
    if st.session_state.conversation:
        if st.button("Stop Conversation", key="stop"):
            st.session_state.conversation = False
            global_context['conversation'] = False
            st.success("Conversation stopped.")
    else:
        if uploaded_file:
            if st.button("Initialize Conversation", key="start"):
                st.session_state.conversation = True
                global_context['conversation'] = True
                st.info("Conversation started. Press 'Stop Conversation' to end.")
                
                # Save the uploaded file
                image_path = f"uploaded_image.{uploaded_file.type.split('/')[-1]}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Encode the image
                base64_image = encode_image(image_path)

                # Analyze the image
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{uploaded_file.type.split('/')[-1]};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                )
                image_analysis = response.choices[0].message.content
                st.success("The image has been analyzed. Now you can start the conversation.")

                # Initialize the conversation context
                global_context['conversation_context'] = [
                    {"role": "system", "content": f"Consider the following image analysis: {image_analysis}. You are my tutor. Guide me through the problem gradually. Never expose the answer directly. Do not answer to anything that is not related to learning subjects. Please respond only in English."},
                    {"role": "assistant", "content": "Hello! I'm here to help you with your learning problem. Let's work through it step-by-step. Please ask your first question."}
                ]

                asyncio.run(start_conversation())

manage_conversation()

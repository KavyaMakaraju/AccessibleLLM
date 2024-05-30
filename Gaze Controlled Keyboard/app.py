import streamlit as st
import requests

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"

# Initialize session state variables
if "message" not in st.session_state:
    st.session_state["message"] = ""

# CSS styles
css = """
<style>
    .title {
        color: #4CAF50;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sidebar-title {
        color: #2196F3;
        font-size: 1.5em;
        margin-bottom: 10px;
    }
    .message {
        color: #FF5722;
        font-size: 1.2em;
        margin-top: 20px;
        white-space: pre-wrap;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        font-size: 1em;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
</style>
"""

# Inject CSS styles
st.markdown(css, unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Gaze-Controlled Keyboard and Text Generation</div>', unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown('<div class="sidebar-title">Controls</div>', unsafe_allow_html=True)

# Function to start the keyboard
def start_keyboard():
    response = requests.post(f"{FASTAPI_URL}/start")
    if response.status_code == 200:
        st.session_state["message"] = "Keyboard started successfully."
    else:
        st.session_state["message"] = "Failed to start keyboard."

# Function to stop the keyboard
def stop_keyboard():
    response = requests.post(f"{FASTAPI_URL}/stop")
    if response.status_code == 200:
        st.session_state["message"] = "Keyboard stopped successfully."
    else:
        st.session_state["message"] = "Failed to stop keyboard."

# Function to get text
def get_text():
    response = requests.get(f"{FASTAPI_URL}/text")
    if response.status_code == 200:
        text_data = response.json()
        st.session_state["message"] = f"Collected Text: {text_data['text']}"
    else:
        st.session_state["message"] = "Failed to get text."

# Function to generate text
def generate_text():
    response = requests.post(f"{FASTAPI_URL}/generate-text")
    if response.status_code == 200:
        generated_texts = response.json()["generated_texts"]
        st.session_state["message"] = "Generated Texts:\n" + "\n".join(
            f"{idx + 1}: {text}" for idx, text in enumerate(generated_texts)
        )
    else:
        st.session_state["message"] = "Failed to generate text."

# Buttons in the sidebar to control the keyboard and text generation
if st.sidebar.button("Start Keyboard"):
    start_keyboard()

if st.sidebar.button("Stop Keyboard"):
    stop_keyboard()

if st.sidebar.button("Get Text"):
    get_text()

if st.sidebar.button("Generate Text"):
    generate_text()

# Display messages in the main content area
if st.session_state["message"]:
    formatted_message = st.session_state["message"].replace("\n", "<br>")
    st.markdown(f'<div class="message">{formatted_message}</div>', unsafe_allow_html=True)

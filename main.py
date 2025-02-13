import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from audio_processor import AudioProcessor
from speech_recognizer import HindiSpeechRecognizer
from utils import plot_waveform
import os

# Set page config
st.set_page_config(
    page_title="Hindi Speech Recognition",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = HindiSpeechRecognizer()

# Create recordings directory if it doesn't exist
if not os.path.exists('recordings'):
    os.makedirs('recordings')

def main():
    st.title("🎤 Hindi Speech Recognition")
    st.markdown("Record your voice in Hindi and get the transcript")

    # Audio recording interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recording Controls")
        if st.button("Start Recording", key="start"):
            st.session_state.recording = True
            st.experimental_rerun()

        if st.button("Stop Recording", key="stop"):
            st.session_state.recording = False
            st.experimental_rerun()

        if st.session_state.get('recording', False):
            st.warning("🔴 Recording in progress...")

    with col2:
        st.subheader("Audio Visualization")
        if st.session_state.get('audio_data') is not None:
            plot_waveform(st.session_state.audio_data)

    # Process recorded audio
    if st.session_state.get('audio_data') is not None:
        st.subheader("Recorded Audio")

        # Save audio file
        filename = f"recordings/recording_{len(os.listdir('recordings'))}.wav"
        sf.write(filename, st.session_state.audio_data, 16000)

        # Process audio
        processed_audio = st.session_state.audio_processor.process(st.session_state.audio_data)

        # Generate transcript
        transcript = st.session_state.recognizer.recognize(processed_audio)

        st.subheader("Transcript")
        st.markdown(f"**Hindi Text:** {transcript}")

        # Download button for audio
        with open(filename, 'rb') as f:
            st.download_button(
                label="Download Recording",
                data=f,
                file_name=filename,
                mime="audio/wav"
            )

if __name__ == "__main__":
    main()

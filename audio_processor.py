import numpy as np
import librosa
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process(self, audio_data):
        """
        Process the audio data for speech recognition
        """
        # Ensure correct sample rate
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(
            audio_data[0],
            audio_data[1:] - pre_emphasis * audio_data[:-1]
        )

        # Apply bandpass filter
        nyquist = self.sample_rate // 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, emphasized_audio)

        # Extract features
        mfccs = librosa.feature.mfcc(
            y=filtered_audio,
            sr=self.sample_rate,
            n_mfcc=13
        )

        return {
            'filtered_audio': filtered_audio,
            'mfccs': mfccs
        }

import sounddevice as sd
import numpy as np
import soundfile as sf


# read wav file
def read_wav_file(file_path):
    """
    Reads a WAV file and returns the audio data and sample rate.
    
    Parameters:
        file_path (str): Path to the WAV file.
        
    Returns:
        tuple: A tuple containing the audio data (numpy array) and sample rate (int).
    """
    try:
        data, fs = sf.read(file_path)
        return data, fs
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None
#
# play the first two chqannels of the wav file

def play_audio(data, fs):
    """
    Plays the audio data using the sounddevice library.
    
    Parameters:
        data (numpy array): Audio data to be played.
        fs (int): Sample rate of the audio data.
    """
    try:
        sd.play(data[:, :2], fs)  # Play only the first two channels
        sd.wait()  # Wait until the sound has finished playing
    except Exception as e:
        print(f"Error playing audio: {e}")
#
# Example usage
if __name__ == "__main__":
    # Define the path to the WAV file
    file_path = "/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/DEMOs/Museum-M/Tests_LOV/SDM/synthesized_scenarios/SDM_Sollazzo_Brou_church_reverb_MM1.wav"
    #"/Users/apple/Library/CloudStorage/Dropbox/My_ESAT/DEMOs/Museum-M/Tests_LOV/SDM/synthesized_scenarios/SDM_Sollazzo_Brou_church_Direct_MM1.wav"  # Replace with your actual file path



    # Read the WAV file
    audio_data, sample_rate = read_wav_file(file_path)

    play_audio(audio_data, sample_rate)
    # Play the audio
    
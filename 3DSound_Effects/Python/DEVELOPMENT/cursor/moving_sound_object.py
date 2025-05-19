import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.spatial import Delaunay
import math
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd
import platform

# Development flag - set to True for testing on laptop speakers
DEVELOPMENT_MODE = True

def print_audio_devices():
    """Print available audio devices for debugging."""
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        print(f"{idx}: {dev['name']} (Input channels: {dev['max_input_channels']}, Output channels: {dev['max_output_channels']})")

def calculate_vbap_gains(speaker_positions, virtual_source_pos):
    """
    Calculate VBAP gains for a virtual source position using the 3 closest speakers.
    """
    # Convert to numpy arrays
    speaker_pos = np.array(speaker_positions)
    source_pos = np.array(virtual_source_pos)
    
    # Calculate distances to all speakers
    distances = np.linalg.norm(speaker_pos - source_pos, axis=1)
    
    # Find 3 closest speakers
    closest_indices = np.argsort(distances)[:3]
    closest_speakers = speaker_pos[closest_indices]
    
    # Create matrix of speaker positions
    L = closest_speakers.T
    
    # Calculate gains using pseudo-inverse
    gains = np.linalg.pinv(L) @ source_pos
    
    # Normalize gains
    gains = gains / np.sum(np.abs(gains))
    
    return gains, closest_indices, distances

def apply_delay(signal, delay_samples, fs):
    """
    Apply fractional delay to a signal using FFT-based resampling.
    """
    # Convert delay in samples to phase shift
    n = len(signal)
    freq = np.fft.rfftfreq(n, 1/fs)
    phase_shift = np.exp(-2j * np.pi * freq * delay_samples/fs)
    
    # Apply phase shift in frequency domain
    signal_fft = np.fft.rfft(signal)
    delayed_fft = signal_fft * phase_shift
    
    # Convert back to time domain
    delayed_signal = np.fft.irfft(delayed_fft, n)
    
    return delayed_signal

def calculate_distance_attenuation(distance, reference_distance=1.0):
    """
    Calculate attenuation based on distance using inverse square law.
    """
    return 1.0 / (1.0 + distance/reference_distance)

def create_moving_sound(duration, fs, speaker_positions, start_speaker_idx, end_speaker_idx):
    """
    Create a sound that moves from one speaker to another using VBAP with time-of-arrival delays.
    """
    try:
        # Load a test sound (you can replace this with your own sound)
        test_sound, _ = sf.read('SDM/dry_excitations/sollazzo_48k.wav')
        print(f"Loaded test sound with shape: {test_sound.shape}")
        
        # Convert to mono if stereo
        if len(test_sound.shape) > 1:
            test_sound = np.mean(test_sound, axis=1)
            print("Converted to mono")
        
        # Calculate number of samples
        num_samples = int(duration * fs)
        print(f"Number of samples: {num_samples}")
        
        # Create output array with extra padding for delays
        num_speakers = len(speaker_positions)
        max_delay_samples = int(2 * fs)  # Maximum delay of 2 seconds
        output = np.zeros((num_samples + max_delay_samples, num_speakers))
        
        # Calculate movement path
        start_pos = speaker_positions[start_speaker_idx]
        end_pos = speaker_positions[end_speaker_idx]
        
        # Create movement path using linear interpolation (simpler than cubic)
        t = np.linspace(0, 1, num_samples)
        path = np.zeros((num_samples, 3))
        for i in range(3):  # x, y, z coordinates
            path[:, i] = start_pos[i] + t * (end_pos[i] - start_pos[i])
        
        # Speed of sound in meters per second
        speed_of_sound = 343.0
        
        # Apply VBAP and delays for each position
        print("Applying VBAP and delays...")
        for i in range(num_samples):
            gains, speaker_indices, distances = calculate_vbap_gains(speaker_positions, path[i])
            
            # Calculate delays for each speaker
            delays = distances / speed_of_sound  # in seconds
            delay_samples = (delays * fs).astype(int)
            
            # Calculate distance-based attenuation
            attenuations = np.array([calculate_distance_attenuation(d) for d in distances])
            
            # Apply gains, delays, and attenuations to the sound
            for j, speaker_idx in enumerate(speaker_indices):
                if i < len(test_sound):
                    # Get the current sample with gain and attenuation
                    current_sample = test_sound[i] * gains[j] * attenuations[speaker_idx]
                    
                    # Calculate the delay for this speaker
                    speaker_delay = delay_samples[speaker_idx]
                    
                    # Apply the delayed sample to the output
                    output[i + speaker_delay, speaker_idx] += current_sample
        
        # Normalize the output to prevent clipping
        output = output / np.max(np.abs(output))
        print("Normalized output")
        
        # Simple low-pass filtering using convolution
        print("Applying simple low-pass filter...")
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        for i in range(num_speakers):
            output[:, i] = np.convolve(output[:, i], kernel, mode='same')
        
        return output
    except Exception as e:
        print(f"Error in create_moving_sound: {e}")
        raise

def main():
    # Print system information
    print(f"Python version: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Running in {'DEVELOPMENT' if DEVELOPMENT_MODE else 'PRODUCTION'} mode")
    
    # Print available audio devices
    print_audio_devices()
    
    # Load speaker positions from CSV file
    try:
        speakers_df = pd.read_csv('layout_8loudspeaker_LOV_GENELEC.csv', sep=',', header=None)
        speaker_positions = speakers_df.iloc[:, :3].values  # Get x, y, z coordinates
        
        print("\nSpeaker positions (x, y, z):")
        for i, pos in enumerate(speaker_positions):
            print(f"Speaker {i+1}: {pos}")
    except Exception as e:
        print(f"Error loading speaker positions: {e}")
        return
    
    # Parameters
    duration = 5.0  # seconds
    fs = 48000  # sample rate
    
    if DEVELOPMENT_MODE:
        # For development, use only 2 speakers (stereo)
        num_speakers = 2
        # Create simple stereo positions
        speaker_positions = np.array([
            [1, 0, 0],    # Left speaker
            [-1, 0, 0]    # Right speaker
        ])
        start_speaker_idx = 0  # Left speaker
        end_speaker_idx = 1    # Right speaker
    else:
        # For production, use all 8 speakers
        num_speakers = len(speaker_positions)
        start_speaker_idx = 1  # Speaker 2
        end_speaker_idx = 5    # Speaker 6
    
    try:
        # Create moving sound
        print("\nCreating moving sound...")
        output = create_moving_sound(duration, fs, speaker_positions, start_speaker_idx, end_speaker_idx)
        print(f"Output shape: {output.shape}")
        print(f"Duration: {output.shape[0]/fs:.2f} seconds")
        
        # Play the sound
        print("\nPlaying sound...")
        sd.play(output, fs)
        sd.wait()
        
        # Save the output
        print("\nSaving output to moving_sound_with_delays.wav...")
        sf.write('moving_sound_with_delays.wav', output, fs)
        print("Done!")
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
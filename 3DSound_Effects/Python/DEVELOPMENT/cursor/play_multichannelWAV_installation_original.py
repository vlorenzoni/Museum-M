import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import pandas as pd
from scipy.spatial import Delaunay
import math
import platform
from scipy import signal
import os

# ===== CONFIGURABLE PARAMETERS =====
# Audio file paths
audio_file = "SDM_Sollazzo_Brou_church.wav"
dry_audiofile = "sollazzo_48k.wav"

# Audio device settings
device_id_or_name = "Digiface USB"  # Options: "Digiface USB", "Dante", "MacBook Pro Speakers"
speakers_NR = 8  # Number of speakers in the layout

# Volume settings
DRY_GAIN = 0.1   # Gain for dry sound
WET_GAIN = 0.1  # Gain for wet sound

# Panning sequence settings
# Format: [(start_time, start_position, end_position, duration), ...]
# Each tuple represents a panning move: at start_time, pan from start_position to end_position over duration seconds
# Positions can be integer speaker numbers (0,1,2) or virtual positions (0.5, 1.5)
PAN_SEQUENCE = [
    (0, 0, 0.5, 3.75),     # At 0s, pan from speaker 0 to virtual position 0.5 in 3.75s
    (3.75, 0.5, 1, 3.75),  # At 3.75s, pan from virtual position 0.5 to speaker 1 in 3.75s
    (7.5, 1, 1.5, 3.75),   # At 7.5s, pan from speaker 1 to virtual position 1.5 in 3.75s
    (11.25, 1.5, 2, 3.75), # At 11.25s, pan from virtual position 1.5 to speaker 2 in 3.75s
    (15, 2, 1.5, 3.75),    # At 15s, pan from speaker 2 to virtual position 1.5 in 3.75s
    (18.75, 1.5, 1, 3.75), # At 18.75s, pan from virtual position 1.5 to speaker 1 in 3.75s
    (22.5, 1, 0.5, 3.75),  # At 22.5s, pan from speaker 1 to virtual position 0.5 in 3.75s
    (26.25, 0.5, 0, 3.75), # At 26.25s, pan from virtual position 0.5 to speaker 0 in 3.75s
]

# Wetness sequence settings (starts after panning)
# Format: [(time, speaker, wetness_percentage), ...]
# Each tuple represents a target wetness at a specific time for a specific speaker
# wetness_percentage: 0.0 = fully dry, 1.0 = fully wet
WETNESS_SEQUENCE = [
    (30, 0, 0.0),    # Start fully dry on speaker 0
    (32, 0, 0.1),    # Very gradual introduction of wet sound on speaker 0
    (49, 0, 1),    # Continue gradual increase on speaker 0
]

# Expansion sequence settings (starts after wetness sequence is complete)
# Format: [(time, center_speaker, spread_radius), ...]
# Each tuple represents when to start expanding from a center speaker
# spread_radius: number of speakers to spread to on each side (0 = no spread, 1 = one speaker each side, etc.)
EXPANSION_SEQUENCE = [
    (55, 0, 0),    # Start with no expansion (just center speaker)
    (60, 0, 1),    # Expand to one speaker on each side
    (65, 0, 2),    # Expand to two speakers on each side
    (70, 0, 3),    # Expand to three speakers on each side
    (75, 0, 2),    # Reduce to two speakers on each side
]

# ===== END OF CONFIGURABLE PARAMETERS =====

def create_smooth_pan(num_channels, fs=48000, max_samples=None):
    """
    Create a smooth panning effect based on the specified panning sequence.
    Supports virtual speaker positions between physical speakers.
    """
    # Find the total duration needed
    total_duration = max([start + duration for start, _, _, duration in PAN_SEQUENCE])
    num_samples = min(int(total_duration * fs), max_samples) if max_samples else int(total_duration * fs)
    gains = np.zeros((num_samples, num_channels))
    
    def get_speaker_gains(position):
        """Calculate gains for a virtual position between speakers"""
        gains = np.zeros(num_channels)
        if position.is_integer():
            gains[int(position)] = 1.0
        else:
            left = int(np.floor(position))
            right = int(np.ceil(position))
            if right >= num_channels:
                right = 0
            right_gain = position - left
            left_gain = 1 - right_gain
            gains[left] = left_gain
            gains[right] = right_gain
        return gains
    
    # Process each panning move
    for start_time, start_pos, end_pos, duration in PAN_SEQUENCE:
        start_sample = min(int(start_time * fs), num_samples)
        end_sample = min(int((start_time + duration) * fs), num_samples)
        if start_sample >= num_samples:
            continue
        
        num_move_samples = end_sample - start_sample
        for i in range(num_move_samples):
            progress = i / num_move_samples
            current_pos = start_pos + (end_pos - start_pos) * progress
            gains[start_sample + i] = get_speaker_gains(current_pos)
    
    return gains

def create_wetness_gains(num_channels, fs=48000, max_samples=None):
    """
    Create wetness gains based on the specified wetness sequence.
    Returns two gain arrays: one for dry sound and one for wet sound.
    """
    total_duration = max([time for time, _, _ in WETNESS_SEQUENCE])
    num_samples = min(int(total_duration * fs), max_samples) if max_samples else int(total_duration * fs)
    dry_gains = np.zeros((num_samples, num_channels))
    wet_gains = np.zeros((num_samples, num_channels))
    
    # Sort wetness sequence by time
    wetness_sequence = sorted(WETNESS_SEQUENCE, key=lambda x: x[0])
    
    # Process each wetness point
    for i in range(len(wetness_sequence)):
        time, speaker, target_wetness = wetness_sequence[i]
        current_sample = min(int(time * fs), num_samples)
        if current_sample >= num_samples:
            continue
        
        if i > 0:
            prev_time, prev_speaker, prev_wetness = wetness_sequence[i-1]
            prev_sample = min(int(prev_time * fs), num_samples)
            
            # Calculate number of samples to interpolate
            num_interp_samples = current_sample - prev_sample
            if num_interp_samples > 0:
                # Create smooth transition using cosine interpolation
                t = np.linspace(0, np.pi, num_interp_samples)
                fade = (1 - np.cos(t)) / 2  # Smooth fade from 0 to 1
                
                for j in range(num_interp_samples):
                    if prev_sample + j >= num_samples:
                        break
                    # Interpolate wetness value
                    current_wetness = prev_wetness + (target_wetness - prev_wetness) * fade[j]
                    # Apply to both dry and wet gains for the target speaker only
                    wet_gains[prev_sample + j, speaker] = current_wetness
                    dry_gains[prev_sample + j, speaker] = 1 - current_wetness
        else:
            # First point - set initial values for the target speaker only
            wet_gains[current_sample:, speaker] = target_wetness
            dry_gains[current_sample:, speaker] = 1 - target_wetness
    
    return dry_gains, wet_gains

def create_expanding_wet_sound(num_channels, fs=48000, max_samples=None):
    """
    Create gains for expanding wet sound based on the expansion sequence.
    Spreads the sound to neighboring speakers with smooth transitions.
    """
    total_duration = max([time for time, _, _ in EXPANSION_SEQUENCE])
    num_samples = min(int(total_duration * fs), max_samples) if max_samples else int(total_duration * fs)
    gains = np.zeros((num_samples, num_channels))
    
    def get_speaker_gains(center_speaker, radius):
        """Calculate gains for speakers around the center speaker"""
        speaker_gains = np.zeros(num_channels)
        radius_int = int(np.round(radius))
        
        for offset in range(-radius_int, radius_int + 1):
            speaker = (center_speaker + offset) % num_channels
            distance = abs(offset)
            if distance <= radius_int:
                # Use cosine window for smooth gain distribution
                gain = np.cos(distance * np.pi / (2 * radius_int)) if radius_int > 0 else 1.0
                speaker_gains[speaker] = gain
        
        # Normalize gains to maintain constant power
        total_gain = np.sum(speaker_gains)
        if total_gain > 0:
            speaker_gains /= total_gain
        
        return speaker_gains
    
    # Sort expansion sequence by time
    expansion_sequence = sorted(EXPANSION_SEQUENCE, key=lambda x: x[0])
    
    # Process each expansion point
    for i in range(len(expansion_sequence)):
        time, center_speaker, radius = expansion_sequence[i]
        current_sample = min(int(time * fs), num_samples)
        if current_sample >= num_samples:
            continue
        
        if i > 0:
            prev_time, prev_center, prev_radius = expansion_sequence[i-1]
            prev_sample = min(int(prev_time * fs), num_samples)
            
            num_interp_samples = current_sample - prev_sample
            if num_interp_samples > 0:
                t = np.linspace(0, np.pi/2, num_interp_samples)
                fade = np.sin(t)  # Smooth transition between radii
                
                for j in range(num_interp_samples):
                    if prev_sample + j >= num_samples:
                        break
                    # Interpolate radius
                    current_radius = prev_radius + (radius - prev_radius) * fade[j]
                    gains[prev_sample + j] = get_speaker_gains(center_speaker, current_radius)
        else:
            gains[current_sample:] = get_speaker_gains(center_speaker, radius)
    
    return gains

def process_audio_chunk(dry_chunk, wet_chunk, pan_gains, dry_gains, wet_gains, expansion_gains, start_idx, chunk_size, speakers_NR):
    """Process a chunk of audio with the given gains"""
    output_chunk = np.zeros((chunk_size, speakers_NR))
    fs = 48000  # Sample rate
    
    # Calculate relative indices within the chunk
    chunk_start = start_idx
    chunk_end = start_idx + chunk_size
    
    # Panning phase (0-30s)
    panning_end = int(30 * fs)
    if chunk_start < panning_end:
        end = min(chunk_end, panning_end) - chunk_start
        for i in range(end):
            rel_idx = chunk_start + i
            if rel_idx < len(pan_gains):
                for ch in range(speakers_NR):
                    output_chunk[i, ch] = dry_chunk[i] * pan_gains[rel_idx, ch] * DRY_GAIN
    
    # Wetness phase (30-55s)
    wetness_start = int(30 * fs)
    wetness_end = int(55 * fs)
    if chunk_start < wetness_end and chunk_end > wetness_start:
        start = max(0, wetness_start - chunk_start)
        end = min(chunk_size, wetness_end - chunk_start)
        for i in range(start, end):
            rel_idx = chunk_start + i
            if rel_idx < len(dry_gains):
                for ch in range(speakers_NR):
                    # Apply both dry and wet gains with proper scaling
                    dry_value = dry_chunk[i] * dry_gains[rel_idx, ch] * DRY_GAIN
                    wet_value = (
                        wet_chunk[i, ch] * wet_gains[rel_idx, ch] * WET_GAIN if len(wet_chunk.shape) > 1 else
                        wet_chunk[i] * wet_gains[rel_idx, ch] * WET_GAIN
                    )
                    output_chunk[i, ch] = dry_value + wet_value
    
    # Expansion phase (55s-end)
    expansion_start = int(55 * fs)
    if chunk_end > expansion_start:
        start = max(0, expansion_start - chunk_start)
        for i in range(start, chunk_size):
            rel_idx = chunk_start + i
            if rel_idx < len(expansion_gains):
                for ch in range(speakers_NR):
                    # Apply expansion gains to wet sound only
                    wet_value = (
                        wet_chunk[i, ch] * expansion_gains[rel_idx, ch] * WET_GAIN if len(wet_chunk.shape) > 1 else
                        wet_chunk[i] * expansion_gains[rel_idx, ch] * WET_GAIN
                    )
                    output_chunk[i, ch] = wet_value
    
    return output_chunk

def main():
    # List available devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        print(f"{idx}: {dev['name']} (Input channels: {dev['max_input_channels']}, Output channels: {dev['max_output_channels']})")

    if device_id_or_name == "Digiface USB":
        for dev in devices:
            if "Digiface" in dev['name']:
                device_id = dev['name']
                break
        gain = 0.02  # Reduced from 0.05 to 0.02
        print(f"Found Digiface device: {device_id}")
        num_channels = speakers_NR

    elif device_id_or_name == "Dante":
        gain = 0.1  # Reduced from 0.3 to 0.1
        print(f"Found Dante device: {device_id_or_name}")
        num_channels = speakers_NR

    elif device_id_or_name == "MacBook Pro Speakers":
        gain = 0.1  # Reduced from 0.3 to 0.1
        num_channels = 2

        if platform.system() == "Darwin":  # Mac
            # Find the default output device
            default_device = sd.query_devices(kind='output')
            device_id = default_device['name']
            gain = 0.1  # Reduced from 0.3 to 0.1
            print(f"No external device found. Using Mac's built-in speakers: {device_id}")
        else:
            raise ValueError("No supported audio device found!")

    # Set the output device with the correct sample rate
    sd.default.device = device_id
    sd.default.channels = num_channels

    # Load speaker configuration only if using multichannel
    if num_channels > 2:
        speakers_df = pd.read_csv('layout_8loudspeaker_LOV_GENELEC.csv', sep=';', header=None)
        # Remove empty rows and get only the first 8 speakers
        speakers_df = speakers_df.iloc[:8, :3]
        speaker_positions = speakers_df.values
        print(f"Number of speakers from CSV: {num_channels}")
        print(f"Speaker positions:\n{speaker_positions}")
        
        # Verify speaker positions are valid
        if np.any(np.isnan(speaker_positions)):
            raise ValueError("Invalid speaker positions detected. Please check the CSV file format.")
    else:
        print("Using stereo output - no speaker positions needed")

    # Load audio files
    dry_audio, fs = sf.read(dry_audiofile)
    wet_audio, fs_wet = sf.read(audio_file)
    
    # Set the sample rate
    sd.default.samplerate = fs
    
    # Check sample rates match
    if fs_wet != fs:
        raise ValueError(f"Sample rate mismatch: dry audio is {fs}Hz, wet audio is {fs_wet}Hz")
    
    # Convert dry audio to mono if needed
    if len(dry_audio.shape) > 1:
        dry_audio = np.mean(dry_audio, axis=1)
    
    # Get audio duration in seconds
    audio_duration = len(dry_audio) / fs
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Ensure both audio files are the same length and properly aligned
    min_length = min(len(dry_audio), len(wet_audio))
    dry_audio = dry_audio[:min_length]
    wet_audio = wet_audio[:min_length, :speakers_NR] if len(wet_audio.shape) > 1 else wet_audio[:min_length]
    
    # Pre-calculate all gains
    print("Calculating gains...")
    pan_gains = create_smooth_pan(speakers_NR, fs, max_samples=min_length)
    dry_gains, wet_gains = create_wetness_gains(speakers_NR, fs, max_samples=min_length)
    expansion_gains = create_expanding_wet_sound(speakers_NR, fs, max_samples=min_length)
    print("Gains calculated, processing audio...")
    
    # Process audio in chunks
    chunk_size = 48000  # Process 1 second at a time
    output = np.zeros((min_length, speakers_NR))
    
    for start_idx in range(0, min_length, chunk_size):
        end_idx = min(start_idx + chunk_size, min_length)
        current_chunk_size = end_idx - start_idx
        
        # Get the current chunks of audio
        dry_chunk = dry_audio[start_idx:end_idx]
        wet_chunk = wet_audio[start_idx:end_idx] if len(wet_audio.shape) == 1 else wet_audio[start_idx:end_idx, :]
        
        # Process the chunk
        output_chunk = process_audio_chunk(
            dry_chunk, wet_chunk, pan_gains, dry_gains, wet_gains, expansion_gains,
            start_idx, current_chunk_size, speakers_NR
        )
        
        # Store the processed chunk
        output[start_idx:end_idx] = output_chunk
    
    print("Audio processing complete, exporting files...")
    
    # Export individual channel files
    for channel in range(speakers_NR):
        output_file = f"speaker_{channel+1}.wav"
        sf.write(output_file, output[:, channel], fs)
        print(f"Exported {output_file}")
    
    print("Starting playback...")
    # Play the audio
    sd.play(output, fs)
    sd.wait()

if __name__ == "__main__":
    main()

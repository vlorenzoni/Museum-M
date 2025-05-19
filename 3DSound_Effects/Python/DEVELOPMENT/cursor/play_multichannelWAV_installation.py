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
import sys
import threading

# ===== CONFIGURABLE PARAMETERS =====
# Audio file paths
audio_file = "SDM_Sollazzo_Brou_church.wav"
dry_audiofile = "sollazzo_48k.wav"

# Audio device settings
device_id_or_name = "Digiface USB"  # Options: "Digiface USB", "Dante", "MacBook Pro Speakers"
speakers_NR = 8  # Number of speakers in the layout

# Volume settings
DRY_GAIN = 0.005   # Reduced from 0.02 to 0.005
WET_GAIN = 0.005  # Reduced from 0.02 to 0.005

# Panning sequence settings
# Format: [(time, position), ...]
# Each tuple represents a target position at a specific time
# Positions can be:
# - integer speaker numbers (0,1,2) for single speaker
# - tuple of speaker numbers ((0,1), (1,2)) for multiple speakers
# - float values (0.5, 1.5) for virtual positions between speakers
PAN_SEQUENCE = [
    (0, 0),          # Start at speaker 0
    (3.75, (0,1)),   # Move to speakers 0 and 1
    (7.5, 1),        # Move to speaker 1
    (11.25, (1,2)),  # Move to speakers 1 and 2
    (15, 2),         # Move to speaker 2
    (18.75, (2,3)),  # Move to speakers 2 and 3
    (22.5, 1),       # Move to speaker 1
    (26.25, (0,1)),  # Move to speakers 0 and 1
    (30, 0),         # Move to speaker 0
]

# Wetness sequence settings (starts after panning)
# Format: [(time, speakers, wetness_percentage), ...]
# Each tuple represents a target wetness at a specific time for the specified speakers
# speakers: can be a single speaker number or a tuple of speaker numbers
# wetness_percentage: 0.0 = fully dry, 1.0 = fully wet
WETNESS_SEQUENCE = [
    (30, (0,1), 0.0),     # Start fully dry on speakers 0 and 1
    (35, (0,1), 0.1),     # Very gradual introduction of wet sound on speakers 0 and 1
    (40, (0,1), 0.5),     # Increase wetness slightly
    (45, (0,1,2), 0.9),   # Expand to three speakers while maintaining wetness
       
]

# Expansion sequence settings (starts after wetness sequence is complete)
# Format: [(time, center_speaker, spread_radius), ...]
# Each tuple represents when to start expanding from a center speaker
# spread_radius: number of speakers to spread to on each side (0 = no spread, 1 = one speaker each side, etc.)
EXPANSION_SEQUENCE = [
    (55, (0,1,2,3), 0),    # Start with no expansion (just center speakers)
    (60, (0,1,2,3), 1),    # Expand to one speaker on each side
    (65, (0,1,2,3), 2),    # Expand to two speakers on each side
    (70, (0,1,2,3), 3),    # Expand to three speakers on each side
    (75, (0,1,2,3), 1),    # Reduce to two speakers on each side
]

# ===== END OF CONFIGURABLE PARAMETERS =====

def create_smooth_pan(num_channels, fs=48000, max_samples=None):
    """
    Create a smooth panning effect based on the specified panning sequence.
    Supports virtual speaker positions between physical speakers and multiple speakers.
    """
    # Find the total duration needed
    total_duration = max([time for time, _ in PAN_SEQUENCE])
    num_samples = min(int(total_duration * fs), max_samples) if max_samples else int(total_duration * fs)
    gains = np.zeros((num_samples, num_channels))
    
    def get_speaker_gains(position):
        """Calculate gains for a virtual position between speakers or multiple speakers"""
        gains = np.zeros(num_channels)
        
        # Handle multiple speaker positions (tuple)
        if isinstance(position, tuple):
            num_speakers = len(position)
            # For constant power, divide by sqrt of number of speakers
            gain_per_speaker = 1.0 / np.sqrt(num_speakers)
            for spk in position:
                gains[spk] = gain_per_speaker
            return gains
            
        # Handle virtual position between speakers
        if isinstance(position, float) and not position.is_integer():
            left = int(np.floor(position))
            right = int(np.ceil(position))
            if right >= num_channels:
                right = 0
            right_gain = position - left
            left_gain = 1 - right_gain
            gains[left] = left_gain
            gains[right] = right_gain
        else:
            # Single speaker
            gains[int(position)] = 1.0
            
        return gains
    
    # Sort panning sequence by time
    pan_sequence = sorted(PAN_SEQUENCE, key=lambda x: x[0])
    
    # Process each panning point
    for i in range(len(pan_sequence)):
        time, position = pan_sequence[i]
        current_sample = min(int(time * fs), num_samples)
        if current_sample >= num_samples:
            continue
        
        if i > 0:
            prev_time, prev_position = pan_sequence[i-1]
            prev_sample = min(int(prev_time * fs), num_samples)
            
            num_interp_samples = current_sample - prev_sample
            if num_interp_samples > 0:
                t = np.linspace(0, np.pi/2, num_interp_samples)
                fade = np.sin(t)  # Smooth transition between positions
                
                for j in range(num_interp_samples):
                    if prev_sample + j >= num_samples:
                        break
                    # Interpolate between previous and current position
                    prev_gains = get_speaker_gains(prev_position)
                    current_gains = get_speaker_gains(position)
                    # Use power-preserving crossfade
                    gains[prev_sample + j] = np.sqrt((1 - fade[j]) * prev_gains**2 + fade[j] * current_gains**2)
        else:
            gains[current_sample:] = get_speaker_gains(position)
    
    return gains

def create_wetness_gains(num_channels, fs=48000, max_samples=None):
    """
    Create wetness gains based on the specified wetness sequence.
    Returns two gain arrays: one for dry sound and one for wet sound.
    Supports multiple speakers with constant power and smooth transitions.
    """
    total_duration = max([time for time, _, _ in WETNESS_SEQUENCE])
    num_samples = min(int(total_duration * fs), max_samples) if max_samples else int(total_duration * fs)
    dry_gains = np.zeros((num_samples, num_channels))
    wet_gains = np.zeros((num_samples, num_channels))
    
    def get_speaker_gains(speakers):
        """Calculate gains for a single speaker or multiple speakers with constant power"""
        gains = np.zeros(num_channels)
        
        # Handle tuple case (multiple speakers)
        if isinstance(speakers, tuple):
            num_speakers = len(speakers)
            # For constant power, divide by sqrt of number of speakers
            gain_per_speaker = 1.0 / np.sqrt(num_speakers)
            for spk in speakers:
                gains[spk] = gain_per_speaker
            return gains
        
        # Handle single speaker
        gains[int(speakers)] = 1.0
        return gains
    
    def apply_crossfade(start_gains, end_gains, start_sample, end_sample, fade_type='cosine'):
        """Apply smooth crossfade between two gain configurations"""
        num_samples = end_sample - start_sample
        if num_samples <= 0:
            return
            
        # Create fade curve
        if fade_type == 'cosine':
            t = np.linspace(0, np.pi, num_samples)
            fade = (1 - np.cos(t)) / 2  # Smooth fade from 0 to 1
        else:
            fade = np.linspace(0, 1, num_samples)  # Linear fade
            
        # Apply crossfade to each channel
        for i in range(num_samples):
            progress = fade[i]
            # Power-preserving crossfade
            gains = np.sqrt((1 - progress) * start_gains**2 + progress * end_gains**2)
            dry_gains[start_sample + i] = gains
            wet_gains[start_sample + i] = gains
    
    # Sort wetness sequence by time
    wetness_sequence = sorted(WETNESS_SEQUENCE, key=lambda x: x[0])
    
    # Process each wetness point
    for i in range(len(wetness_sequence)):
        time, speakers, target_wetness = wetness_sequence[i]
        current_sample = min(int(time * fs), num_samples)
        if current_sample >= num_samples:
            continue
        
        if i > 0:
            prev_time, prev_speakers, prev_wetness = wetness_sequence[i-1]
            prev_sample = min(int(prev_time * fs), num_samples)
            
            # Calculate number of samples to interpolate
            num_interp_samples = current_sample - prev_sample
            if num_interp_samples > 0:
                # Get speaker gains for both start and end positions
                prev_gains = get_speaker_gains(prev_speakers)
                current_gains = get_speaker_gains(speakers)
                
                # First crossfade the speaker configuration
                apply_crossfade(prev_gains, current_gains, prev_sample, current_sample)
                
                # Then apply wetness changes with a separate crossfade
                t = np.linspace(0, np.pi, num_interp_samples)
                wetness_fade = (1 - np.cos(t)) / 2  # Smooth fade from 0 to 1
                
                for j in range(num_interp_samples):
                    if prev_sample + j >= num_samples:
                        break
                    # Interpolate wetness value
                    current_wetness = prev_wetness + (target_wetness - prev_wetness) * wetness_fade[j]
                    
                    # Calculate total gain for normalization
                    total_gain = np.sqrt((1 - current_wetness)**2 + current_wetness**2)
                    if total_gain > 0:
                        # Apply normalized gains
                        gains = dry_gains[prev_sample + j]
                        dry_gains[prev_sample + j] = gains * ((1 - current_wetness) / total_gain)
                        wet_gains[prev_sample + j] = gains * (current_wetness / total_gain)
        else:
            # First point - set initial values
            speaker_gains = get_speaker_gains(speakers)
            # Calculate total gain for normalization
            total_gain = np.sqrt((1 - target_wetness)**2 + target_wetness**2)
            if total_gain > 0:
                # Apply normalized gains
                dry_gains[current_sample:] = speaker_gains * ((1 - target_wetness) / total_gain)
                wet_gains[current_sample:] = speaker_gains * (target_wetness / total_gain)
    
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
        """Calculate gains for speakers around the center speaker(s)"""
        speaker_gains = np.zeros(num_channels)
        radius_int = int(np.round(radius))
        
        # Handle tuple of center speakers
        if isinstance(center_speaker, tuple):
            for center in center_speaker:
                for offset in range(-radius_int, radius_int + 1):
                    speaker = (center + offset) % num_channels
                    distance = abs(offset)
                    if distance <= radius_int:
                        # Use cosine window for smooth gain distribution
                        gain = np.cos(distance * np.pi / (2 * radius_int)) if radius_int > 0 else 1.0
                        speaker_gains[speaker] = max(speaker_gains[speaker], gain)
        else:
            # Handle single center speaker
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

def display_timeline(current_time):
    """Display current time and active effects in a simple format"""
    time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
    effects = get_current_effect_info(current_time)
    
    # Clear the current line
    print("\r", end="")
    
    # Print simplified timeline
    if current_time < 30:
        print(f"Time: {time_str} - Panning sequence - Phase 1", end="")
    elif current_time < 55:
        print(f"Time: {time_str} - Wetness sequence - Phase 2", end="")
    else:
        print(f"Time: {time_str} - Expansion sequence - Phase 3", end="")
    
    sys.stdout.flush()

def get_current_effect_info(current_time):
    """Get information about the currently active effects"""
    info = []
    
    # Check panning sequence
    for time, position in PAN_SEQUENCE:
        if time <= current_time:
            if isinstance(position, tuple):
                info.append(f"speakers {position}")
            else:
                info.append(f"speaker {position}")
    
    # Check wetness sequence
    for time, speakers, wetness in WETNESS_SEQUENCE:
        if time <= current_time:
            if isinstance(speakers, tuple):
                info.append(f"wetness {wetness*100:.0f}% on speakers {speakers}")
            else:
                info.append(f"wetness {wetness*100:.0f}% on speaker {speakers}")
    
    # Check expansion sequence
    for time, center_speaker, radius in EXPANSION_SEQUENCE:
        if time <= current_time:
            if isinstance(center_speaker, tuple):
                info.append(f"expanding from speakers {center_speaker} with radius {radius}")
            else:
                info.append(f"expanding from speaker {center_speaker} with radius {radius}")
    
    return info

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

def display_timeline_thread():
    """Thread function to display timeline during playback"""
    while True:
        current_time = time.time() - start_time
        if current_time < 30:
            print(f"\rTime: {int(current_time//60):02d}:{int(current_time%60):02d} - Panning sequence - Phase 1", end="")
        elif current_time < 55:
            print(f"\rTime: {int(current_time//60):02d}:{int(current_time%60):02d} - Wetness sequence - Phase 2", end="")
        else:
            print(f"\rTime: {int(current_time//60):02d}:{int(current_time%60):02d} - Expansion sequence - Phase 3", end="")
        time.sleep(0.1)  # Update every 100ms

def main():
    # Create audio_output directory if it doesn't exist
    output_dir = "audio_output"
    os.makedirs(output_dir, exist_ok=True)
    
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
        gain = 0.005
        print(f"Found Digiface device: {device_id}")
        num_channels = speakers_NR

    elif device_id_or_name == "Dante":
        gain = 0.005
        print(f"Found Dante device: {device_id_or_name}")
        num_channels = speakers_NR

    elif device_id_or_name == "MacBook Pro Speakers":
        gain = 0.005
        num_channels = 2

        if platform.system() == "Darwin":  # Mac
            default_device = sd.query_devices(kind='output')
            device_id = default_device['name']
            gain = 0.005
            print(f"No external device found. Using Mac's built-in speakers: {device_id}")
        else:
            raise ValueError("No supported audio device found!")

    # Set the output device with the correct sample rate
    sd.default.device = device_id
    sd.default.channels = num_channels
    sd.default.samplerate = 48000  # Explicitly set sample rate

    # Load audio files
    dry_audio, fs = sf.read(dry_audiofile)
    wet_audio, fs_wet = sf.read(audio_file)
    
    # Check sample rates match
    if fs_wet != fs:
        raise ValueError(f"Sample rate mismatch: dry audio is {fs}Hz, wet audio is {fs_wet}Hz")
    
    # Convert dry audio to mono if needed
    if len(dry_audio.shape) > 1:
        dry_audio = np.mean(dry_audio, axis=1)
    
    # Get audio duration in seconds
    audio_duration = len(dry_audio) / fs
    print(f"\nAudio duration: {audio_duration:.2f} seconds")
    print("Processing audio...")
    
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
    
    print("\nAudio processing complete, saving files...")
    
    # Export individual channel files to audio_output folder
    for channel in range(speakers_NR):
        output_file = os.path.join(output_dir, f"speaker_{channel+1}.wav")
        sf.write(output_file, output[:, channel], fs)
        print(f"Saved {output_file}")
    
    print("\nStarting playback...")
    
    # Normalize the output to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val
    
    # Start timeline display thread
    global start_time
    start_time = time.time()
    timeline_thread = threading.Thread(target=display_timeline_thread)
    timeline_thread.daemon = True  # Thread will be killed when main program exits
    timeline_thread.start()
    
    # Play the audio with increased volume
    sd.play(output * gain, fs)
    sd.wait()
    
    print("\nPlayback complete!")

if __name__ == "__main__":
    main()

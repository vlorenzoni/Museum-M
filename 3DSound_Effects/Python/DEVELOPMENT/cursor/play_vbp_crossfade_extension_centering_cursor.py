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

# Define audio file paths
# get file two folders up
audio_file = Path(__file__).resolve().parents[2] / "SDM_Sollazzo_Brou_church.wav"

dry_audiofile = Path(__file__).resolve().parents[2] / "sollazzo_48k.wav"
#device_id_or_name = "Digiface USB"
device_id_or_name = "Dante"
# device_id_or_name = "Dante"  # "DANTE"
speakers_NR = 8  # Number of speakers in the layout to be modified
speakers_csv = Path(__file__).resolve().parents[2] /"layout_8loudspeaker_LOV_GENELEC.csv"  # CSV file with speaker positions



def calculate_vbap_gains(speaker_positions, virtual_source_pos, fs=48000):
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
    
    return gains, closest_indices

def create_smooth_pan(duration, num_channels, fs=48000):
    """
    Create a smooth panning effect for stereo or multichannel output.
    For stereo: pans between left and right
    For multichannel: pans through all speakers
    """
    num_samples = int(duration * fs)
    gains = np.zeros((num_samples, num_channels))
    
    if num_channels == 2:  # Stereo case
        # Create a smooth pan between left and right
        pan = np.linspace(0, 1, num_samples)  # 0 = left, 1 = right
        gains[:, 0] = 1 - pan  # Left channel
        gains[:, 1] = pan      # Right channel
    else:  # Multichannel case
        # Calculate time per speaker
        samples_per_speaker = num_samples // num_channels
        
        for i in range(num_channels):
            start_sample = i * samples_per_speaker
            end_sample = (i + 1) * samples_per_speaker
            
            # Create smooth transition between speakers
            if i < num_channels - 1:
                # Fade out current speaker
                fade_out = np.linspace(1, 0, samples_per_speaker)
                gains[start_sample:end_sample, i] = fade_out
                
                # Fade in next speaker
                fade_in = np.linspace(0, 1, samples_per_speaker)
                gains[start_sample:end_sample, i + 1] = fade_in
            else:
                # Last speaker stays on
                gains[start_sample:end_sample, i] = 1
    
    return gains

def create_expanding_wet_sound(duration, num_channels, fs=48000):
    """
    Create gains for gradually expanding wet sound.
    For stereo: fades in right channel
    For multichannel: expands to neighboring speakers
    """
    num_samples = int(duration * fs)
    gains = np.zeros((num_samples, num_channels))
    
    if num_channels == 2:  # Stereo case
        # Start with both channels at full volume
        gains[:, 0] = 1
        gains[:, 1] = 1
    else:  # Multichannel case
        # Start with last speaker
        last_speaker = num_channels - 1
        gains[:, last_speaker] = 1
        
        # Expand every 5 seconds
        expansion_interval = 5  # seconds
        samples_per_expansion = int(expansion_interval * fs)
        num_expansions = int(duration // expansion_interval)
        
        for i in range(1, num_expansions + 1):
            start_sample = i * samples_per_expansion
            if start_sample >= num_samples:
                break
                
            # Add speakers to the right and left
            right_speaker = (last_speaker + i) % num_channels
            left_speaker = (last_speaker - i) % num_channels
            
            # Calculate end sample, ensuring we don't exceed array bounds
            end_sample = min(start_sample + samples_per_expansion, num_samples)
            current_samples = end_sample - start_sample
            
            # Create fade in for the current expansion
            fade_in = np.linspace(0, 1, current_samples)
            
            # Apply fade in to new speakers
            gains[start_sample:end_sample, right_speaker] = fade_in
            gains[start_sample:end_sample, left_speaker] = fade_in
            
            # Maintain previous speakers
            gains[end_sample:, right_speaker] = 1
            gains[end_sample:, left_speaker] = 1
    
    return gains


def create_center_motion_effect(duration_sec, speaker_positions, fs=48000):
    """
    From t = 50s to t = 70s, move sound from periphery toward center and back.
    Uses smoother transitions with minimum gain threshold to prevent complete speaker cutoffs.
    """
    num_samples = int(duration_sec * fs)
    effect = np.zeros((num_samples, len(speaker_positions)))

    # Define motion points: edge → center → edge
    center = np.mean(speaker_positions, axis=0)
    radius = np.max(np.linalg.norm(speaker_positions - center, axis=1))
    
    # Create smoother motion curve with minimum gain threshold
    times = np.linspace(0, 1, num_samples)
    # Modified triangle shape with minimum value of 0.2 instead of 0
    motion_curve = 0.2 + 0.8 * (1 - np.abs(2 * times - 1))  

    for i in range(num_samples):
        r = motion_curve[i] * radius
        # Generate a circular path around the center with smoother angular motion
        angle = 2 * np.pi * times[i]
        pos = center + r * np.array([np.cos(angle), np.sin(angle), 0])
        gains, indices = calculate_vbap_gains(speaker_positions, pos)
        
        # Apply minimum gain threshold to prevent complete cutoffs
        gains = np.maximum(gains, 0.1)
        # Renormalize gains after applying threshold
        gains = gains / np.sum(gains)
        
        for g, idx in zip(gains, indices):
            effect[i, idx] += g

    return effect


# List available devices
print("\nAvailable audio devices:")
devices = sd.query_devices()

for idx, dev in enumerate(devices):
    print(f"{idx}: {dev['name']} (Input channels: {dev['max_input_channels']}, Output channels: {dev['max_output_channels']})")


if device_id_or_name is "Digiface USB":
    device_id_or_name = dev['name']
    gain = 0.1
    print(f"Found Digiface device: {device_id_or_name}")
    num_channels = speakers_NR

elif device_id_or_name is "Dante":
    gain = 1.0
    print(f"Found Dante device: {device_id_or_name}")
    num_channels = speakers_NR

elif device_id_or_name is "MacBook Pro Speakers":
    gain = 1.0
    num_channels = 2

    if platform.system() == "Darwin":  # Mac
        # Find the default output device
        default_device = sd.query_devices(kind='output')
        device_id_or_name = default_device['name']
        gain = 1.0
        print(f"No external device found. Using Mac's built-in speakers: {device_id_or_name}")
    else:
        raise ValueError("No supported audio device found!")
    
        

# Load speaker configuration only if using multichannel
if num_channels > 2:
    speakers_df = pd.read_csv(speakers_csv, sep=';', header=None)
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

# Load the stimulus file first to get sample rate
SDMaudio, fs = sf.read(audio_file)
print(f"SDM audio shape: {SDMaudio.shape}, Sample rate: {fs}")

# Load dry audio file
dry_audio, fs_dry = sf.read(dry_audiofile)
if fs_dry != fs:
    raise ValueError(f"Sample rate mismatch: SDM audio is {fs}Hz, dry audio is {fs_dry}Hz")
print(f"Dry audio shape: {dry_audio.shape}, Sample rate: {fs}")

# Set the output device with the correct sample rate
sd.default.device = device_id_or_name
sd.default.channels = num_channels
sd.default.samplerate = fs

# Convert stereo to mono if needed and apply gain
if len(dry_audio.shape) > 1:
    dry_audio = np.mean(dry_audio, axis=1)  # Average the channels
dry_audio = dry_audio.reshape(-1) * gain  # Apply gain after converting to mono
print(f"Dry audio after conversion: {dry_audio.shape}, max amplitude: {np.max(np.abs(dry_audio))}")

# Process SDM audio
if len(SDMaudio.shape) > 1:
    if num_channels == 2:  # If using stereo
        # Convert 8 channels to stereo by taking front left and right channels
        SDMaudio = SDMaudio[:, [0, 1]] * gain  # Use first two channels
    else:
        SDMaudio = SDMaudio[:,:num_channels] * gain
else:
    if num_channels == 2:  # If using stereo
        SDMaudio = np.tile(SDMaudio.reshape(-1, 1), (1, 2)) * gain
    else:
        SDMaudio = np.tile(SDMaudio.reshape(-1, 1), (1, num_channels)) * gain

print(f"Processed SDM audio shape: {SDMaudio.shape}")

# Update duration calculations (around line 290)
dry_pan_duration = 24  # seconds
crossfade_duration = 5  # seconds
wet_duration = 41  # seconds (increased to accommodate longer center motion)
center_motion_start = 50  # seconds
center_motion_duration = 20  # seconds (until 70s)
total_duration = max(center_motion_start + center_motion_duration, 
                    dry_pan_duration + crossfade_duration + wet_duration)

samples_dry = int(dry_pan_duration * fs)
samples_crossfade = int(crossfade_duration * fs)
samples_wet = int(wet_duration * fs)

# Create output array with the correct number of channels (ensure it's long enough)
output = np.zeros((int(total_duration * fs), num_channels), dtype=np.float32)

# Create dry pan gains
dry_pan_gains = create_smooth_pan(dry_pan_duration, num_channels, fs)
print(f"Dry pan gains shape: {dry_pan_gains.shape}")

# Create expanding wet gains
wet_gains = create_expanding_wet_sound(wet_duration, num_channels, fs)
print(f"Wet gains shape: {wet_gains.shape}")

# Update the center motion effect application (around line 310)
print(f"\nApplying center motion effect from {center_motion_start}s to {center_motion_start + center_motion_duration}s")
center_motion_start_sample = int(center_motion_start * fs)
center_motion_end_sample = center_motion_start_sample + int(center_motion_duration * fs)

if center_motion_end_sample <= output.shape[0]:
    effect_gains = create_center_motion_effect(center_motion_duration, speaker_positions, fs)
    # Use SDMaudio for the effect instead of dry_audio for better spatial effect
    effect_audio = SDMaudio[center_motion_start_sample:center_motion_start_sample + len(effect_gains)]
    if len(effect_audio) < len(effect_gains):
        effect_audio = np.pad(effect_audio, ((0, len(effect_gains) - len(effect_audio)), (0, 0)))
    
    # Apply the center motion effect
    for i in range(effect_gains.shape[0]):
        output[center_motion_start_sample + i] = effect_audio[i] * effect_gains[i]
else:
    print("Warning: Center motion effect duration exceeds output buffer.")

# Ensure we have enough audio data
if len(dry_audio) < samples_dry + samples_crossfade:
    print("Warning: Dry audio is shorter than needed, repeating...")
    dry_audio = np.tile(dry_audio, int(np.ceil((samples_dry + samples_crossfade) / len(dry_audio))))
    dry_audio = dry_audio[:samples_dry + samples_crossfade]

if len(SDMaudio) < samples_crossfade + samples_wet:
    print("Warning: Wet audio is shorter than needed, repeating...")
    SDMaudio = np.tile(SDMaudio, int(np.ceil((samples_crossfade + samples_wet) / len(SDMaudio))))
    SDMaudio = SDMaudio[:samples_crossfade + samples_wet]

print(f"Final audio lengths - Dry: {len(dry_audio)}, Wet: {len(SDMaudio)}")

# Apply dry pan
for i in range(samples_dry):
    output[i] = dry_audio[i] * dry_pan_gains[i]

# Create crossfade
fade_out = np.linspace(1, 0, samples_crossfade)
fade_in = np.linspace(0, 1, samples_crossfade)

# Apply crossfade - both dry and wet use the same playback position
for i in range(samples_crossfade):
    pos = samples_dry + i
    # During crossfade, wet sound starts at the last speaker position
    wet_gain = wet_gains[0]  # Use initial wet gain (last speaker only)
    output[pos] = (dry_audio[pos] * dry_pan_gains[-1] * fade_out[i] + 
                  SDMaudio[pos] * wet_gain * fade_in[i])  # Use same position for wet audio

# Apply expanding wet sound after crossfade
start_wet = samples_dry + samples_crossfade
for i in range(samples_wet):
    pos = start_wet + i
    if pos < len(SDMaudio):
        output[pos] = SDMaudio[pos] * wet_gains[i]

print(f"Dry audio length: {len(dry_audio)} samples ({len(dry_audio)/fs:.2f} seconds)")
print(f"Wet audio length: {len(SDMaudio)} samples ({len(SDMaudio)/fs:.2f} seconds)")
print(f"Output shape: {output.shape} ({len(output)/fs:.2f} seconds)")

# Update sync markers (around line 380)
def add_sync_markers(audio, fs, marker_duration=0.1, marker_amplitude=0.5):
    """
    Add synchronization markers at specific timestamps.
    Returns the audio with markers and a list of marker positions.
    """
    markers = {
        'dry_start': 0,
        'crossfade_start': 24,
        'wet_start': 29,
        'wet_expansion_1': 34,
        'wet_expansion_2': 39,
        'wet_expansion_3': 44,
        'center_motion_start': 50,
        'center_motion_mid': 60,  # Midpoint of center motion
        'center_motion_end': 70
    }
    
    # Create marker audio
    marker_samples = int(marker_duration * fs)
    marker = np.ones((marker_samples, audio.shape[1])) * marker_amplitude
    
    # Add markers to audio
    for name, position in markers.items():
        start_sample = int(position * fs)
        end_sample = start_sample + marker_samples
        if end_sample < len(audio):
            audio[start_sample:end_sample] = marker
    
    return audio, markers

# Add markers to the output
output_with_markers, marker_positions = add_sync_markers(output.copy(), fs)

# Save the final mixed file with proper metadata and markers
metadata = {
    'artist': 'Museum Installation',
    'title': 'Spatial Audio Mix',
    'channels': num_channels,
    'samplerate': fs,
    'duration': total_duration,
    'markers': marker_positions
}

sf.write('installation_audio.wav', output_with_markers, fs, format='WAV', subtype='FLOAT')
print(f"Saved installation audio with shape: {output_with_markers.shape}")
print(f"Duration: {total_duration} seconds")
print(f"Sample rate: {fs} Hz")
print(f"Channels: {num_channels}")
print("\nSynchronization markers at:")
for name, position in marker_positions.items():
    print(f"- {name}: {position} seconds")

# Create a simple README file with instructions
with open('README.txt', 'w') as f:
    f.write("""Installation Audio Setup Instructions:

1. Audio File: installation_audio.wav
   - Duration: {} seconds
   - Sample Rate: {} Hz
   - Channels: {}

2. Audio Timeline:
   0-24s:  Dry panning phase
   24-29s: Crossfade phase
   29-50s: Wet sound expansion
   50-70s: Center motion effect
           (50-60s: Moving toward center)
           (60-70s: Moving back to periphery)
   70s+:   Continued wet sound

3. Playback Instructions:
   - Use VLC Media Player
   - Set to loop continuously
   - Configure audio device for {} channels
   - Match speaker layout with channel mapping

4. Auto-start Setup:
   - Use the appropriate startup script for your OS:
     * Mac: start_installation_mac.sh
     * Windows: start_installation_windows.bat
     * Linux: start_installation_linux.sh
   - Add script to system startup programs

Note: Ensure audio device is configured for {} channels before starting playback.
""".format(total_duration, fs, num_channels, num_channels, num_channels))

print("\nCreated README.txt with installation instructions")

print("\nPlaying final mix...")

def audio_callback(outdata, frames, time, status):
    """
    Callback function for real-time audio playback with progress display.
    """
    if status:
        print(status)
    
    global current_frame
    if current_frame < len(output):
        chunk = output[current_frame:current_frame + frames]
        if len(chunk) < frames:
            chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)))
        outdata[:] = chunk
        current_frame += frames
        
        current_time = current_frame / fs
        if current_time < dry_pan_duration:
            phase = "Dry panning"
            progress = current_time / dry_pan_duration * 100
        elif current_time < dry_pan_duration + crossfade_duration:
            phase = "Crossfade"
            progress = (current_time - dry_pan_duration) / crossfade_duration * 100
        elif current_time >= center_motion_start and current_time < center_motion_start + center_motion_duration:
            phase = "Center Motion"
            progress = (current_time - center_motion_start) / center_motion_duration * 100
            motion_stage = "Moving inward" if progress < 50 else "Moving outward"
            print(f"\r{phase} - {motion_stage}: {progress:.1f}% complete (Time: {current_time:.1f}s)", 
                  end="", flush=True)
            return
        else:
            phase = "Wet sound"
            progress = 100
        
        print(f"\rTime: {current_time:.1f}s / {total_duration:.1f}s - {phase} ({progress:.1f}%)", 
              end="", flush=True)
    else:
        raise sd.CallbackStop

try:
    # Create a stream with callback
    current_frame = 0
    with sd.OutputStream(
        samplerate=fs,
        channels=num_channels,
        dtype='float32',
        device=device_id_or_name,
        callback=audio_callback,
        blocksize=1024  # Adjust this value to control update frequency
    ) as stream:
        print("Playback started")
        stream.start()
        while stream.active:
            time.sleep(0.1)
        print("\nPlayback finished!")
except Exception as e:
    print(f"Error during playback: {e}")
finally:
    sd.stop()

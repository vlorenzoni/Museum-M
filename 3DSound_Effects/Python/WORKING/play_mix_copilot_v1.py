import sounddevice as sd
import soundfile as sf
import numpy as np
import time


# Example usage

audio_file = "SDM_Sollazzo_Brou_church.wav"  # Replace with your .wav file path

# Load the stimulus file
SDMaudio, fs = sf.read(audio_file)

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())


## SELECT  DEVICE and change number of channels
# 0 = Dante, 6 = DIGIFACE 
device_id_or_name = 6 

if device_id_or_name == 0:
    gain = 1
elif device_id_or_name == 6:
    gain = .08

num_channels = 8 #sd.query_devices(device_id_or_name)['max_output_channels'] # change this to number of speakers
SDMaudio = SDMaudio[:,:num_channels]

# print length of SDMaudio  
print(f"Length of SDMaudio: {len(SDMaudio)} samples ({len(SDMaudio)/fs:.2f} seconds)")


# SDMaudio = SDMaudio[:45 * 48000]

print(f"Using device: {device_id_or_name}")
print(f"Loaded audio with shape: {SDMaudio.shape}, Sample rate: {fs}")

#play each midi channel to a separate speaker AT THE SAME TIME

# === List Devices and Choose One ===
print("Available output devices with at least 8 channels:")
for idx, dev in enumerate(sd.query_devices()):
    if dev['max_output_channels'] >= 8:
        print(f"{idx}: {dev['name']} ({dev['max_output_channels']} channels)")

# === Play audio: channel 0 → speaker 1, ..., channel 7 → speaker 8 ===

# use only 8 channels of digiface
# Set the output device
sd.default.device = device_id_or_name  # Set the output device to the selected one
# Set the number of output channels
sd.default.channels = num_channels  # Set the number of output channels
# Set the sample rate
sd.default.samplerate = fs  # Set the sample rate
# Play the audio
print(f"Playing audio on device {device_id_or_name} with {num_channels} channels")
# Play the audio

#lower the gains of the channels

SDMaudio = SDMaudio * gain

SDMaudio_test = SDMaudio[:10 * fs]  # First 5 seconds
sd.play(SDMaudio_test, fs)
sd.wait()


#split all the channels in separate wav files
for i in range(SDMaudio.shape[1]):
    sf.write(f"channel_{i}.wav",SDMaudio[:, i], fs)
    
#    # Generate filename and save individual microphone channels

# Yes, sd.play() plays all channels simultaneously
# We can verify the simultaneous playback by checking the audio array shapes    
print("\nPlayback details:")
print(f"Number of channels playing simultaneously: {SDMaudio.shape[1]}")
print("Each column in the audio array represents a separate channel")
print("All channels are played in sync through sd.play()")

# Optional: Add a visual confirmation of active channels
print("\nChannel activity:")
for ch in range(SDMaudio.shape[1]):
    # Check if channel contains non-zero audio
    if np.any(SDMaudio[:, ch] != 0):
        print(f"Channel {ch}: Active")
    else:
        print(f"Channel {ch}: Silent")


## create a mix

   #load dry_excitations/sollazzo_48k.wav
dry_audiofile = "sollazzo_48k.wav"
dry_audio, fs = sf.read(dry_audiofile)

dry_audio = dry_audio * gain



print(f"Length of dry audio: {len(dry_audio)} samples ({len(dry_audio)/fs:.2f} seconds)")
print(f"Loaded audio with shape: {dry_audio.shape}, Sample rate: {fs}")

#play dry audio
test_dry_audio = dry_audio[:10 * fs]  # First 5 seconds
sd.play(test_dry_audio, fs)
sd.wait()


## define mixing parameters

first_effect_duration = 15 # seconds

crossfade_duration = 5 # seconds
second_effect_duration = 5 # seconds


# Calculate durations
samples_first = int(first_effect_duration * fs)  # First section length
samples_crossfade = int(crossfade_duration * fs)    # Crossfade length
samples_second= samples_first + samples_crossfade  # Point where crossfade ends
total_samples = samples_first + samples_crossfade + samples_second  # Total length: 15s + 5s crossfade + 20s


# Prepare dry audio
if len(dry_audio.shape) > 1:
    dry_audio = dry_audio[:, 0]
dry_audio = dry_audio.reshape(-1)
dry_repeated = np.tile(dry_audio[:samples_first + samples_crossfade].reshape(-1, 1), (1, num_channels))

# Create fade curves
fade_out = np.linspace(1, 0, samples_crossfade)
fade_in = np.linspace(0, 1, samples_crossfade)


output = np.zeros((total_samples, num_channels))

# 1. first effect

output[:samples_first] = dry_repeated[:samples_first]

# 2. crossfade
for i in range(samples_crossfade):
    pos = samples_first + i
    output[pos] = (dry_repeated[pos] * fade_out[i] + 
                  SDMaudio[pos] * fade_in[i])

# 3. second effect
output[samples_second:] = SDMaudio[samples_first + samples_crossfade :total_samples]

# Save and play
sf.write('final_mix.wav', output, fs)

sd.play(output, fs)
# Display time while playing
total_duration = len(output) / fs
start_time = time.time()

try:
    while sd.get_stream().active:
        current_time = time.time() - start_time
        if current_time < first_effect_duration:
            print(" - First effect")
        elif current_time < first_effect_duration + crossfade_duration:
            print(" - Crossfade")
        else:
            print(" - Second effect")
        # Print elapsed time
        print(f"\rTime: {current_time:0.1f}s / {total_duration:0.1f}s", end="", flush=True)
        time.sleep(0.1)
    print("\nPlayback finished!")
except KeyboardInterrupt:
    sd.stop()
    print("\nPlayback stopped by user")
